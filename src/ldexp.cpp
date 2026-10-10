// SPDX-License-Identifier: MIT

#include "ldexp.h"
#include "alu.h"
#include "dpp.h"
#include "fp64.h"
#include "goc/goc.h"
#include "internal.h"

#include <cmath>
#include <cstring>
#include <stdint.h>

namespace {

template <bool Fp64>
[[gnu::noinline]] int ldexp_guest(uint64_t flags, uint32_t exec_mask, uint32_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b);

template <bool Fp64, bool GuestDenormals = false>
int ldexp(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
          const uint32_t *const *a, const uint32_t *const *b) {
  const uint32_t known = GOC_ALU_NEG_A | GOC_ALU_ABS_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP;
  if (int error = goc::validate(flags, mode & ~known, false,
                                GOC_FP_IEEE_MODE | GOC_FP_FLUSH_INPUT_DENORMALS |
                                    GOC_FP_FLUSH_OUTPUT_DENORMALS))
    return error;
  if (exec_mask == 0)
    return GOC_SUCCESS;
  if constexpr (!GuestDenormals)
    if (flags & (GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS))
      return ldexp_guest<Fp64>(flags, exec_mask, mode, d, a, b);
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    if constexpr (GuestDenormals)
      goc::ldexp_guest_x86_64_v4(Fp64, flags, exec_mask, mode, d, a, b[0]);
    else
      goc::ldexp_x86_64_v4(Fp64, exec_mask, mode, d, a, b[0]);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    if constexpr (GuestDenormals)
      goc::ldexp_guest_x86_64_v3(Fp64, flags, exec_mask, mode, d, a, b[0]);
    else
      goc::ldexp_x86_64_v3(Fp64, exec_mask, mode, d, a, b[0]);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[Fp64 ? 2 : 1][32];
  for (int lane = 0; lane < 32; ++lane) {
    int32_t exponent;
    std::memcpy(&exponent, b[0] + lane, sizeof(exponent));
    if constexpr (Fp64) {
      double input = goc::fp64_input(a, lane, mode);
      if (GuestDenormals && (flags & GOC_FP_FLUSH_INPUT_DENORMALS))
        input = goc::fp64_flush_denorm(input);
      double value = std::ldexp(input, exponent);
      if ((mode & GOC_ALU_OMOD_HALF) && std::abs(value) == 0x1p-1022) {
        // A tiny exact result can round up to minimum normal. Detect this
        // from the source exponent without relying on wider host FP types.
        int source_exponent;
        std::frexp(input, &source_exponent);
        if (int64_t(source_exponent) + exponent <= -1022)
          value = 0;
      }
      value = goc::fp64_output(value, mode);
      if (GuestDenormals && (flags & GOC_FP_FLUSH_OUTPUT_DENORMALS))
        value = goc::fp64_flush_denorm(value);
      uint64_t bits = goc::double_bits(value);
      result[0][lane] = uint32_t(bits);
      result[1][lane] = uint32_t(bits >> 32);
    } else {
      float input = goc::alu_input(a[0][lane], mode);
      if (GuestDenormals && (flags & GOC_FP_FLUSH_INPUT_DENORMALS))
        input = goc::flush_denorm_f32(input);
      float value;
      if (mode & GOC_ALU_OMOD_HALF) {
        // OMOD flushes tininess before FP32 rounding, including values
        // that would round upward to minimum normal.
        double wide = std::ldexp(double(input), exponent);
        value = std::abs(wide) < 0x1p-126 ? 0.0f : float(wide);
      } else {
        value = std::ldexp(input, exponent);
      }
      value = goc::alu_output_f32(value, mode);
      if (GuestDenormals && (flags & GOC_FP_FLUSH_OUTPUT_DENORMALS))
        value = goc::flush_denorm_f32(value);
      result[0][lane] = goc::as_bits(value);
    }
  }
  for (int reg = 0; reg < (Fp64 ? 2 : 1); ++reg)
    for (int lane = 0; lane < 32; ++lane)
      if ((exec_mask >> lane) & 1)
        d[reg][lane] = result[reg][lane];
  return GOC_SUCCESS;
}

// Keep guest denormal handling out of the default scalar loop. Inlining it
// allows the compiler to merge both loops and reintroduce per-lane flag checks.
template <bool Fp64>
[[gnu::noinline]] int ldexp_guest(uint64_t flags, uint32_t exec_mask, uint32_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b) {
  return ldexp<Fp64, true>(flags, exec_mask, mode, d, a, b);
}

template <bool Fp64>
int ldexp_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                 const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  constexpr auto instruction = Fp64 ? goc_v_ldexp_f64_rdna4 : goc_v_ldexp_f32_rdna4;
  if (int error = instruction(flags, 0, mode, nullptr, nullptr, nullptr, excp_flag_user))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
  for (unsigned half = 0; half < 2; ++half) {
    uint64_t half_mode = mode;
    if (mode & GOC_DPP16) {
      uint64_t rows = (mode & GOC_DPP_ROW_MASK) >> GOC_DPP_ROW_SHIFT;
      half_mode = (mode & ~GOC_DPP_ROW_MASK) | (((rows >> (half * 2)) & 3) << GOC_DPP_ROW_SHIFT);
    }
    uint32_t *dest[Fp64 ? 2 : 1];
    const uint32_t *source[Fp64 ? 2 : 1];
    for (unsigned reg = 0; reg < (Fp64 ? 2U : 1U); ++reg) {
      dest[reg] = d[reg] + half * 32;
      source[reg] = a[reg] + half * 32;
    }
    // The signed integer exponent occupies one VGPR even for FP64.
    const uint32_t *exponent = b[0] + half * 32;
    if (int error = instruction(flags, uint32_t(exec_mask >> (half * 32)), half_mode, dest, source,
                                &exponent, excp_flag_user))
      return error;
  }
  return GOC_SUCCESS;
}

} // namespace

int goc_v_ldexp_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return goc::execute_dpp(flags, exec_mask, mode, a,
                            [&](uint32_t exec_mask, const uint32_t *const *source) {
                              return ldexp<false>(flags, exec_mask, uint32_t(mode), d, source, b);
                            });
  return ldexp<false>(flags, exec_mask, mode, d, a, b);
}

int goc_v_ldexp_f64_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return ldexp<true>(flags, exec_mask, mode, d, a, b);
}

int goc_v_ldexp_f32_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, uint32_t *excp_flag_user) {
  return ldexp_wave64<false>(flags, exec_mask, mode, d, a, b, excp_flag_user);
}

int goc_v_ldexp_f64_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, uint32_t *excp_flag_user) {
  return ldexp_wave64<true>(flags, exec_mask, mode, d, a, b, excp_flag_user);
}

int goc_v_ldexp_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                    const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts) {
  // RDNA3 gates OMOD on guest IEEE mode and output-denormal preservation.
  if (!(flags & GOC_FP_FLUSH_OUTPUT_DENORMALS) || (flags & GOC_FP_IEEE_MODE))
    mode &= ~uint64_t(GOC_ALU_OMOD_HALF);
  return goc_v_ldexp_f32_rdna4(flags, exec_mask, mode, d, a, b, trapsts);
}

int goc_v_ldexp_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts) {
  // RDNA3 gates OMOD on guest IEEE mode and output-denormal preservation.
  if (!(flags & GOC_FP_FLUSH_OUTPUT_DENORMALS) || (flags & GOC_FP_IEEE_MODE))
    mode &= ~uint64_t(GOC_ALU_OMOD_HALF);
  return goc_v_ldexp_f32_rdna4_wave64(flags, exec_mask, mode, d, a, b, trapsts);
}

int goc_v_ldexp_f64(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                    const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts) {
  // RDNA3 gates OMOD on guest IEEE mode and output-denormal preservation.
  if (!(flags & GOC_FP_FLUSH_OUTPUT_DENORMALS) || (flags & GOC_FP_IEEE_MODE))
    mode &= ~uint64_t(GOC_ALU_OMOD_HALF);
  return goc_v_ldexp_f64_rdna4(flags, exec_mask, mode, d, a, b, trapsts);
}

int goc_v_ldexp_f64_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts) {
  // RDNA3 gates OMOD on guest IEEE mode and output-denormal preservation.
  if (!(flags & GOC_FP_FLUSH_OUTPUT_DENORMALS) || (flags & GOC_FP_IEEE_MODE))
    mode &= ~uint64_t(GOC_ALU_OMOD_HALF);
  return goc_v_ldexp_f64_rdna4_wave64(flags, exec_mask, mode, d, a, b, trapsts);
}

// SPDX-License-Identifier: MIT

#include "half_exponent.h"
#include "alu.h"
#include "dpp.h"
#include "float_formats.h"
#include "goc/goc.h"
#include "internal.h"

#include <algorithm>
#include <cmath>
#include <stdint.h>

namespace {

[[gnu::noinline]] int ldexp_guest(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b);

template <bool Ldexp, bool GuestDenormals = false>
int run(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b) {
  if (mode >> 32)
    return goc::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return run<Ldexp, GuestDenormals>(flags, exec_mask, uint32_t(mode), d, source, b);
        });
  const uint32_t known = GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP |
                         GOC_ALU_HIGH_A | GOC_ALU_HIGH_D | (Ldexp ? GOC_ALU_HIGH_B : 0);
  if (int error = goc::validate(flags, mode & ~known, false,
                                GOC_FP_IEEE_MODE | GOC_FP_FLUSH_INPUT_DENORMALS |
                                    GOC_FP_FLUSH_OUTPUT_DENORMALS))
    return error;
  if (exec_mask == 0)
    return GOC_SUCCESS;
  if constexpr (Ldexp && !GuestDenormals)
    if (flags & (GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS))
      return ldexp_guest(flags, exec_mask, mode, d, a, b);
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    if constexpr (GuestDenormals)
      goc::half_ldexp_guest_x86_64_v3(flags, exec_mask, mode, d[0], a[0], b[0]);
    else
      goc::half_exponent_x86_64_v3<Ldexp>(bool(flags & GOC_FP16_OVFL), exec_mask, mode, d[0], a[0],
                                          Ldexp ? b[0] : nullptr);
    return GOC_SUCCESS;
  }
#endif
  int a_shift = mode & GOC_ALU_HIGH_A ? 16 : 0;
  int b_shift = mode & GOC_ALU_HIGH_B ? 16 : 0;
  int d_shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  uint16_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    uint16_t bits = uint16_t(a[0][lane] >> a_shift);
    if constexpr (Ldexp) {
      if (GuestDenormals && (flags & GOC_FP_FLUSH_INPUT_DENORMALS) && (bits & 0x7fff) < 0x400)
        bits &= 0x8000;
      uint16_t raw_exponent = uint16_t(b[0][lane] >> b_shift);
      int exponent = int(raw_exponent) - (raw_exponent & 0x8000 ? 65536 : 0);
      // Following rocjitsu's F16 LDEXP model, bound the exponent while retaining
      // every FP16 rounding outcome, including output scaling and saturation.
      float value = std::ldexp(goc::alu_input(goc::as_bits(goc::f16_to_float(bits)), mode),
                               std::clamp(exponent, -64, 64));
      if (mode & GOC_ALU_OMOD_HALF) {
        // Test tininess before rounding; finite overflow is rounded before OMOD.
        value = std::abs(value) < 0x1p-14f
                    ? 0.0f
                    : goc::f16_to_float(goc::float_to_f16(value, flags & GOC_FP16_OVFL));
      }
      result[lane] = goc::float_to_f16(goc::alu_output_f16(value, mode), flags & GOC_FP16_OVFL);
      if (GuestDenormals && (flags & GOC_FP_FLUSH_OUTPUT_DENORMALS) &&
          (result[lane] & 0x7fff) < 0x400)
        result[lane] &= 0x8000;
    } else {
      // Sign modifiers do not affect the exponent; OMOD is ignored for integer
      // results and all FP16 exponents fit int16_t, so CLAMP has no effect.
      int field = (bits >> 10) & 31, fraction = bits & 1023;
      int exponent = field - 14;
      if (field == 31 || (bits & 0x7fff) == 0)
        exponent = 0;
      else if (field == 0) {
        exponent = -24;
        while (fraction) {
          ++exponent;
          fraction >>= 1;
        }
      }
      result[lane] = uint16_t(exponent);
    }
  }
  for (int lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = (d[0][lane] & ~(0xffffU << d_shift)) | (uint32_t(result[lane]) << d_shift);
  return GOC_SUCCESS;
}

[[gnu::noinline]] int ldexp_guest(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b) {
  // Separate entry prevents loop merging from adding guest checks to default arithmetic.
  return run<true, true>(flags, exec_mask, mode, d, a, b);
}

template <bool Ldexp>
int run_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b) {
  if (int error = run<Ldexp>(flags, 0, mode, nullptr, nullptr, nullptr))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
  for (unsigned half = 0; half < 2; ++half) {
    uint64_t half_mode = mode;
    if (mode & GOC_DPP16) {
      uint64_t rows = (mode & GOC_DPP_ROW_MASK) >> GOC_DPP_ROW_SHIFT;
      half_mode = (mode & ~GOC_DPP_ROW_MASK) | (((rows >> (half * 2)) & 3) << GOC_DPP_ROW_SHIFT);
    }
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *source = a[0] + half * 32;
    const uint32_t *exponent = Ldexp ? b[0] + half * 32 : nullptr;
    if (int error = run<Ldexp>(flags, uint32_t(exec_mask >> (half * 32)), half_mode, &dest, &source,
                               &exponent))
      return error;
  }
  return GOC_SUCCESS;
}

} // namespace

int goc_v_ldexp_f16_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<true>(flags, exec_mask, mode, d, a, b);
}

int goc_v_frexp_exp_i16_f16_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<false>(flags, exec_mask, mode, d, a, nullptr);
}

int goc_v_ldexp_f16_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run_wave64<true>(flags, exec_mask, mode, d, a, b);
}

int goc_v_frexp_exp_i16_f16_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run_wave64<false>(flags, exec_mask, mode, d, a, nullptr);
}

int goc_v_ldexp_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                    const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts) {
  // RDNA3 gates floating output scaling on guest IEEE/denormal mode.
  if (!(flags & GOC_FP_FLUSH_OUTPUT_DENORMALS) || (flags & GOC_FP_IEEE_MODE))
    mode &= ~uint64_t(GOC_ALU_OMOD_HALF);
  return goc_v_ldexp_f16_rdna4(flags, exec_mask, mode, d, a, b, trapsts);
}

int goc_v_ldexp_f16_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts) {
  // RDNA3 gates floating output scaling on guest IEEE/denormal mode.
  if (!(flags & GOC_FP_FLUSH_OUTPUT_DENORMALS) || (flags & GOC_FP_IEEE_MODE))
    mode &= ~uint64_t(GOC_ALU_OMOD_HALF);
  return goc_v_ldexp_f16_rdna4_wave64(flags, exec_mask, mode, d, a, b, trapsts);
}

int goc_v_frexp_exp_i16_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, uint32_t *trapsts) {
  return goc_v_frexp_exp_i16_f16_rdna4(flags, exec_mask, mode, d, a, trapsts);
}

int goc_v_frexp_exp_i16_f16_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *trapsts) {
  return goc_v_frexp_exp_i16_f16_rdna4_wave64(flags, exec_mask, mode, d, a, trapsts);
}

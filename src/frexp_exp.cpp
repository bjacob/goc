// SPDX-License-Identifier: MIT

#include "frexp_exp.h"
#include "bits.h"
#include "dpp.h"
#include "goc/goc.h"
#include "internal.h"

#include <stdint.h>

namespace {

template <bool Fp64>
int frexp_exp(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
              const uint32_t *const *a) {
  const uint32_t known = GOC_ALU_NEG_A | GOC_ALU_ABS_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP;
  if (int error = goc::validate(flags, mode & ~known, !Fp64,
                                Fp64 ? 0
                                     : GOC_FP_IEEE_MODE | GOC_FP_FLUSH_INPUT_DENORMALS |
                                           GOC_FP_FLUSH_OUTPUT_DENORMALS))
    return error;
  if (exec_mask == 0)
    return GOC_SUCCESS;
  // Sign modifiers do not affect the exponent. Integer results ignore OMOD;
  // the exponent always fits in int32_t, so CLAMP does not change the result.
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::frexp_exp_x86_64_v3(Fp64, exec_mask, d[0], a, flags & GOC_FP_FLUSH_INPUT_DENORMALS);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    // Match rocjitsu's frexp exponent model, including zero for NaNs/infinities.
    int exponent = 0;
    if constexpr (Fp64) {
      uint64_t magnitude = (uint64_t(a[0][lane]) | (uint64_t(a[1][lane]) << 32)) & INT64_MAX;
      unsigned field = unsigned(magnitude >> 52);
      if (field && field != 2047)
        exponent = int(field) - 1022;
      else if (!field && magnitude)
        // The leading bit is 63-clz; a subnormal exponent is that position-1073.
        exponent = -1010 - int(goc::bit_count_zero<false>(magnitude));
    } else {
      uint32_t magnitude = a[0][lane] & 0x7fffffffu;
      unsigned field = magnitude >> 23;
      if (field && field != 255)
        exponent = int(field) - 126;
      else if (!field && magnitude && !(flags & GOC_FP_FLUSH_INPUT_DENORMALS))
        // rocjitsu's frexp_f32: exponent = -125 - (countl_zero(magnitude) - 8).
        exponent = -117 - int(goc::bit_count_zero<false>(magnitude));
    }
    result[lane] = uint32_t(exponent);
  }
  for (int lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

// Keep DPP's routing buffers off the ordinary instruction's call path.
[[gnu::noinline]] int frexp_exp_f32_dpp(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                        uint32_t *const *d, const uint32_t *const *a) {
  return goc::execute_dpp(flags, exec_mask, mode, a,
                          [&](uint32_t exec_mask, const uint32_t *const *source) {
                            return frexp_exp<false>(flags, exec_mask, uint32_t(mode), d, source);
                          });
}

} // namespace

int goc_v_frexp_exp_i32_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  // Both architectures use the same numeric model; RDNA3 status is not yet verified.
  return goc_v_frexp_exp_i32_f32_rdna4(flags, exec_mask, mode, d, a, nullptr);
}

int goc_v_frexp_exp_i32_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  uint32_t *excp_flag_user) {
  const auto semantics = flags & GOC_SEMANTICS_MASK;
  if (excp_flag_user && semantics != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return frexp_exp_f32_dpp(flags, exec_mask, mode, d, a);
  return frexp_exp<false>(flags, exec_mask, mode, d, a);
}

int goc_v_frexp_exp_i32_f64(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return frexp_exp<true>(flags, exec_mask, mode, d, a);
}

namespace {

template <bool Fp64, bool Rdna4 = false>
int frexp_exp_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                     const uint32_t *const *a, uint32_t *excp_flag_user) {
  const auto instruction = Fp64    ? goc_v_frexp_exp_i32_f64
                           : Rdna4 ? goc_v_frexp_exp_i32_f32_rdna4
                                   : goc_v_frexp_exp_i32_f32;
  if (int error = instruction(flags, 0, mode, nullptr, nullptr, excp_flag_user))
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
    const uint32_t *source[Fp64 ? 2 : 1];
    for (unsigned reg = 0; reg < (Fp64 ? 2U : 1U); ++reg)
      source[reg] = a[reg] + half * 32;
    if (int error = instruction(flags, uint32_t(exec_mask >> (half * 32)), half_mode, &dest, source,
                                excp_flag_user))
      return error;
  }
  return GOC_SUCCESS;
}

} // namespace

int goc_v_frexp_exp_i32_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *trapsts) {
  return frexp_exp_wave64<false>(flags, exec_mask, mode, d, a, trapsts);
}

int goc_v_frexp_exp_i32_f32_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         uint32_t *excp_flag_user) {
  return frexp_exp_wave64<false, true>(flags, exec_mask, mode, d, a, excp_flag_user);
}

int goc_v_frexp_exp_i32_f64_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  return frexp_exp_wave64<true>(flags, exec_mask, mode, d, a, excp_flag_user);
}

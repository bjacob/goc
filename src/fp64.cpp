// SPDX-License-Identifier: MIT

#include "fp64.h"
#include "goc/goc.h"
#include "internal.h"
#include "minmax.h"

#include <cmath>
#include <stdint.h>

namespace {

// Return instruction modifiers with OMOD disabled when RDNA3 guest mode requires it.
uint64_t rdna3_output_modifiers(uint64_t flags, uint64_t mode) {
  return !(flags & GOC_FP_FLUSH_OUTPUT_DENORMALS) || (flags & GOC_FP_IEEE_MODE)
             ? mode & ~uint64_t(GOC_ALU_OMOD_HALF)
             : mode;
}

template <goc::Fp64 Op, bool Rdna3 = false>
int arithmetic(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  uint32_t known = GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP;
  if constexpr (goc::fp64_sources(Op) >= 2)
    known |= GOC_ALU_ABS_B | GOC_ALU_NEG_B;
  if constexpr (goc::fp64_sources(Op) == 3)
    known |= GOC_ALU_ABS_C | GOC_ALU_NEG_C;
  constexpr bool basic = Op == goc::Fp64::Add || Op == goc::Fp64::Mul || Op == goc::Fp64::Fma;
  constexpr bool selection = Op == goc::Fp64::MinLegacy || Op == goc::Fp64::MaxLegacy ||
                             Op == goc::Fp64::MinNum || Op == goc::Fp64::MaxNum ||
                             Op == goc::Fp64::Minimum || Op == goc::Fp64::Maximum;
  constexpr uint64_t fp_flags =
      GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS | GOC_FP_IEEE_MODE;
  if (int error = goc::validate(flags, mode & ~known, false, fp_flags))
    return error;
  if constexpr (Rdna3)
    mode = uint32_t(rdna3_output_modifiers(flags, mode));
  if (exec_mask == 0)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::fp64_x86_64_v3(Op, flags, exec_mask, mode, d, a, b, c);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[2][32];
  for (int lane = 0; lane < 32; ++lane) {
    double x = goc::fp64_input(a, lane, mode), y = 0;
    if constexpr (goc::fp64_sources(Op) >= 2)
      y = goc::fp64_input(b, lane, mode >> 1);
    if ((basic || selection) && (flags & GOC_FP_FLUSH_INPUT_DENORMALS)) {
      x = goc::fp64_flush_denorm(x);
      y = goc::fp64_flush_denorm(y);
    }
    double value;
    if constexpr (Op == goc::Fp64::Add)
      value = x + y;
    if constexpr (Op == goc::Fp64::Mul)
      value = x * y;
    if constexpr (Op == goc::Fp64::Fma) {
      double z = goc::fp64_input(c, lane, mode >> 2);
      if (flags & GOC_FP_FLUSH_INPUT_DENORMALS)
        z = goc::fp64_flush_denorm(z);
      value = std::fma(x, y, z);
    }
    if constexpr (Op == goc::Fp64::MinLegacy || Op == goc::Fp64::MaxLegacy)
      value = goc::minmax_rdna3<Op == goc::Fp64::MaxLegacy>(x, y, flags & GOC_FP_IEEE_MODE);
    if constexpr (Op == goc::Fp64::MinNum)
      value = goc::minmax<false, false>(x, y);
    if constexpr (Op == goc::Fp64::MaxNum)
      value = goc::minmax<true, false>(x, y);
    if constexpr (Op == goc::Fp64::Minimum)
      value = goc::minmax<false, true>(x, y);
    if constexpr (Op == goc::Fp64::Maximum)
      value = goc::minmax<true, true>(x, y);
    if constexpr (Op == goc::Fp64::FrexpMant) {
      uint64_t magnitude = goc::double_bits(x) & 0x7fffffffffffffffULL;
      int exponent;
      value = magnitude == 0 || magnitude >= 0x7ff0000000000000ULL ? x : std::frexp(x, &exponent);
    }
    if constexpr (Op == goc::Fp64::Trunc)
      value = std::trunc(x);
    if constexpr (Op == goc::Fp64::Ceil)
      value = std::ceil(x);
    if constexpr (Op == goc::Fp64::Rndne)
      value = goc::fp64_rndne(x);
    if constexpr (Op == goc::Fp64::Floor)
      value = std::floor(x);
    if constexpr (Op == goc::Fp64::Sqrt)
      value = std::sqrt(x);
    if constexpr (Op == goc::Fp64::Rcp)
      value = 1.0 / x;
    if constexpr (Op == goc::Fp64::Rsq)
      value = 1.0 / std::sqrt(x);
    if constexpr (Op == goc::Fp64::Fract) {
      value = x - std::floor(x);
      double limit = goc::as_double(0x3fefffffffffffffULL);
      if (value > limit)
        value = limit;
    }
    value = goc::fp64_output(value, mode);
    if (basic && (flags & GOC_FP_FLUSH_OUTPUT_DENORMALS))
      value = goc::fp64_flush_denorm(value);
    uint64_t bits = goc::double_bits(value);
    result[0][lane] = uint32_t(bits);
    result[1][lane] = uint32_t(bits >> 32);
  }
  // Stage both halves before any destination write, including cross-half aliases.
  for (int reg = 0; reg < 2; ++reg)
    for (int lane = 0; lane < 32; ++lane)
      if ((exec_mask >> lane) & 1)
        d[reg][lane] = result[reg][lane];
  return GOC_SUCCESS;
}

template <goc::Fp64 Op, bool Rdna3 = false>
int arithmetic_wave64(uint64_t flags, uint64_t exec_mask, uint32_t mode, uint32_t *const *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      const uint32_t *const *c) {
  if (int error = arithmetic<Op, Rdna3>(flags, 0, mode, nullptr, nullptr, nullptr, nullptr))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
  for (unsigned half = 0; half < 2; ++half) {
    uint32_t *dest[] = {d[0] + half * 32, d[1] + half * 32};
    const uint32_t *x[] = {a[0] + half * 32, a[1] + half * 32};
    const uint32_t *y[2] = {}, *z[2] = {};
    if constexpr (goc::fp64_sources(Op) >= 2)
      for (unsigned word = 0; word < 2; ++word)
        y[word] = b[word] + half * 32;
    if constexpr (goc::fp64_sources(Op) == 3)
      for (unsigned word = 0; word < 2; ++word)
        z[word] = c[word] + half * 32;
    if (int error =
            arithmetic<Op, Rdna3>(flags, uint32_t(exec_mask >> (half * 32)), mode, dest, x, y, z))
      return error;
  }
  return GOC_SUCCESS;
}

} // namespace

int goc_v_add_f64(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic<goc::Fp64::Add, true>(flags, exec_mask, mode, d, a, b, nullptr);
}

int goc_v_add_f64_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic<goc::Fp64::Add>(flags, exec_mask, mode, d, a, b, nullptr);
}

int goc_v_mul_f64(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic<goc::Fp64::Mul, true>(flags, exec_mask, mode, d, a, b, nullptr);
}

int goc_v_mul_f64_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic<goc::Fp64::Mul>(flags, exec_mask, mode, d, a, b, nullptr);
}

int goc_v_fma_f64(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
                  uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic<goc::Fp64::Fma, true>(flags, exec_mask, mode, d, a, b, c);
}

int goc_v_fma_f64_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic<goc::Fp64::Fma>(flags, exec_mask, mode, d, a, b, c);
}

int goc_v_trunc_f64_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                          const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic<goc::Fp64::Trunc>(flags, exec_mask, mode, d, a, nullptr, nullptr);
}

int goc_v_ceil_f64_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic<goc::Fp64::Ceil>(flags, exec_mask, mode, d, a, nullptr, nullptr);
}

int goc_v_rndne_f64_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                          const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic<goc::Fp64::Rndne>(flags, exec_mask, mode, d, a, nullptr, nullptr);
}

int goc_v_floor_f64_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                          const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic<goc::Fp64::Floor>(flags, exec_mask, mode, d, a, nullptr, nullptr);
}

int goc_v_fract_f64_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                          const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic<goc::Fp64::Fract>(flags, exec_mask, mode, d, a, nullptr, nullptr);
}

int goc_v_sqrt_f64_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic<goc::Fp64::Sqrt>(flags, exec_mask, mode, d, a, nullptr, nullptr);
}

int goc_v_rcp_f64_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                        const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic<goc::Fp64::Rcp>(flags, exec_mask, mode, d, a, nullptr, nullptr);
}

int goc_v_rsq_f64_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                        const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic<goc::Fp64::Rsq>(flags, exec_mask, mode, d, a, nullptr, nullptr);
}

int goc_v_min_num_f64(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic<goc::Fp64::MinNum>(flags, exec_mask, mode, d, a, b, nullptr);
}

int goc_v_max_num_f64(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic<goc::Fp64::MaxNum>(flags, exec_mask, mode, d, a, b, nullptr);
}

int goc_v_minimum_f64(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic<goc::Fp64::Minimum>(flags, exec_mask, mode, d, a, b, nullptr);
}

int goc_v_maximum_f64(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic<goc::Fp64::Maximum>(flags, exec_mask, mode, d, a, b, nullptr);
}

int goc_v_frexp_mant_f64_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic<goc::Fp64::FrexpMant>(flags, exec_mask, mode, d, a, nullptr, nullptr);
}

int goc_v_min_f64(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic<goc::Fp64::MinLegacy, true>(flags, exec_mask, mode, d, a, b, nullptr);
}

int goc_v_max_f64(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic<goc::Fp64::MaxLegacy, true>(flags, exec_mask, mode, d, a, b, nullptr);
}

int goc_v_add_f64_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic_wave64<goc::Fp64::Add, true>(flags, exec_mask, mode, d, a, b, nullptr);
}

int goc_v_add_f64_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic_wave64<goc::Fp64::Add>(flags, exec_mask, mode, d, a, b, nullptr);
}

int goc_v_mul_f64_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic_wave64<goc::Fp64::Mul, true>(flags, exec_mask, mode, d, a, b, nullptr);
}

int goc_v_mul_f64_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic_wave64<goc::Fp64::Mul>(flags, exec_mask, mode, d, a, b, nullptr);
}

int goc_v_fma_f64_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic_wave64<goc::Fp64::Fma, true>(flags, exec_mask, mode, d, a, b, c);
}

int goc_v_fma_f64_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c,
                               uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic_wave64<goc::Fp64::Fma>(flags, exec_mask, mode, d, a, b, c);
}

int goc_v_trunc_f64_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic_wave64<goc::Fp64::Trunc>(flags, exec_mask, mode, d, a, nullptr, nullptr);
}

int goc_v_ceil_f64_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a,
                                uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic_wave64<goc::Fp64::Ceil>(flags, exec_mask, mode, d, a, nullptr, nullptr);
}

int goc_v_rndne_f64_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic_wave64<goc::Fp64::Rndne>(flags, exec_mask, mode, d, a, nullptr, nullptr);
}

int goc_v_floor_f64_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic_wave64<goc::Fp64::Floor>(flags, exec_mask, mode, d, a, nullptr, nullptr);
}

int goc_v_fract_f64_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic_wave64<goc::Fp64::Fract>(flags, exec_mask, mode, d, a, nullptr, nullptr);
}

int goc_v_sqrt_f64_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a,
                                uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic_wave64<goc::Fp64::Sqrt>(flags, exec_mask, mode, d, a, nullptr, nullptr);
}

int goc_v_rcp_f64_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic_wave64<goc::Fp64::Rcp>(flags, exec_mask, mode, d, a, nullptr, nullptr);
}

int goc_v_rsq_f64_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic_wave64<goc::Fp64::Rsq>(flags, exec_mask, mode, d, a, nullptr, nullptr);
}

int goc_v_min_num_f64_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic_wave64<goc::Fp64::MinNum>(flags, exec_mask, mode, d, a, b, nullptr);
}

int goc_v_max_num_f64_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic_wave64<goc::Fp64::MaxNum>(flags, exec_mask, mode, d, a, b, nullptr);
}

int goc_v_minimum_f64_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic_wave64<goc::Fp64::Minimum>(flags, exec_mask, mode, d, a, b, nullptr);
}

int goc_v_maximum_f64_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic_wave64<goc::Fp64::Maximum>(flags, exec_mask, mode, d, a, b, nullptr);
}

int goc_v_frexp_mant_f64_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                      uint32_t *const *d, const uint32_t *const *a,
                                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic_wave64<goc::Fp64::FrexpMant>(flags, exec_mask, mode, d, a, nullptr, nullptr);
}

int goc_v_min_f64_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic_wave64<goc::Fp64::MinLegacy, true>(flags, exec_mask, mode, d, a, b, nullptr);
}

int goc_v_max_f64_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic_wave64<goc::Fp64::MaxLegacy, true>(flags, exec_mask, mode, d, a, b, nullptr);
}

int goc_v_trunc_f64(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                    const uint32_t *const *a, uint32_t *trapsts) {
  return goc_v_trunc_f64_rdna4(flags, exec_mask, rdna3_output_modifiers(flags, mode), d, a,
                               trapsts);
}

int goc_v_trunc_f64_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, uint32_t *trapsts) {
  return goc_v_trunc_f64_rdna4_wave64(flags, exec_mask, rdna3_output_modifiers(flags, mode), d, a,
                                      trapsts);
}

int goc_v_ceil_f64(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a, uint32_t *trapsts) {
  return goc_v_ceil_f64_rdna4(flags, exec_mask, rdna3_output_modifiers(flags, mode), d, a, trapsts);
}

int goc_v_ceil_f64_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                          const uint32_t *const *a, uint32_t *trapsts) {
  return goc_v_ceil_f64_rdna4_wave64(flags, exec_mask, rdna3_output_modifiers(flags, mode), d, a,
                                     trapsts);
}

int goc_v_rndne_f64(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                    const uint32_t *const *a, uint32_t *trapsts) {
  return goc_v_rndne_f64_rdna4(flags, exec_mask, rdna3_output_modifiers(flags, mode), d, a,
                               trapsts);
}

int goc_v_rndne_f64_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, uint32_t *trapsts) {
  return goc_v_rndne_f64_rdna4_wave64(flags, exec_mask, rdna3_output_modifiers(flags, mode), d, a,
                                      trapsts);
}

int goc_v_floor_f64(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                    const uint32_t *const *a, uint32_t *trapsts) {
  return goc_v_floor_f64_rdna4(flags, exec_mask, rdna3_output_modifiers(flags, mode), d, a,
                               trapsts);
}

int goc_v_floor_f64_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, uint32_t *trapsts) {
  return goc_v_floor_f64_rdna4_wave64(flags, exec_mask, rdna3_output_modifiers(flags, mode), d, a,
                                      trapsts);
}

int goc_v_fract_f64(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                    const uint32_t *const *a, uint32_t *trapsts) {
  return goc_v_fract_f64_rdna4(flags, exec_mask, rdna3_output_modifiers(flags, mode), d, a,
                               trapsts);
}

int goc_v_fract_f64_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, uint32_t *trapsts) {
  return goc_v_fract_f64_rdna4_wave64(flags, exec_mask, rdna3_output_modifiers(flags, mode), d, a,
                                      trapsts);
}

int goc_v_sqrt_f64(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a, uint32_t *trapsts) {
  return goc_v_sqrt_f64_rdna4(flags, exec_mask, rdna3_output_modifiers(flags, mode), d, a, trapsts);
}

int goc_v_sqrt_f64_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                          const uint32_t *const *a, uint32_t *trapsts) {
  return goc_v_sqrt_f64_rdna4_wave64(flags, exec_mask, rdna3_output_modifiers(flags, mode), d, a,
                                     trapsts);
}

int goc_v_rcp_f64(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, uint32_t *trapsts) {
  return goc_v_rcp_f64_rdna4(flags, exec_mask, rdna3_output_modifiers(flags, mode), d, a, trapsts);
}

int goc_v_rcp_f64_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, uint32_t *trapsts) {
  return goc_v_rcp_f64_rdna4_wave64(flags, exec_mask, rdna3_output_modifiers(flags, mode), d, a,
                                    trapsts);
}

int goc_v_rsq_f64(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, uint32_t *trapsts) {
  return goc_v_rsq_f64_rdna4(flags, exec_mask, rdna3_output_modifiers(flags, mode), d, a, trapsts);
}

int goc_v_rsq_f64_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, uint32_t *trapsts) {
  return goc_v_rsq_f64_rdna4_wave64(flags, exec_mask, rdna3_output_modifiers(flags, mode), d, a,
                                    trapsts);
}

int goc_v_frexp_mant_f64(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, uint32_t *trapsts) {
  return goc_v_frexp_mant_f64_rdna4(flags, exec_mask, rdna3_output_modifiers(flags, mode), d, a,
                                    trapsts);
}

int goc_v_frexp_mant_f64_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return goc_v_frexp_mant_f64_rdna4_wave64(flags, exec_mask, rdna3_output_modifiers(flags, mode), d,
                                           a, trapsts);
}

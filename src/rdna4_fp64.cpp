// SPDX-License-Identifier: MIT

#include "rdna4_fp64.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_minmax.h"

#include <cmath>
#include <stdint.h>

namespace {

template <goc::Fp64 Op>
int arithmetic(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  uint32_t known = GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP;
  if constexpr (goc::fp64_sources(Op) >= 2)
    known |= GOC_ALU_ABS_B | GOC_ALU_NEG_B;
  if constexpr (goc::fp64_sources(Op) == 3)
    known |= GOC_ALU_ABS_C | GOC_ALU_NEG_C;
  if (int error = goc::validate(flags, mode & ~known))
    return error;
  if (exec_mask == 0)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::fp64_x86_64_v3(Op, exec_mask, mode, d, a, b, c);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[2][32];
  for (int lane = 0; lane < 32; ++lane) {
    double x = goc::fp64_input(a, lane, mode), y = 0;
    if constexpr (goc::fp64_sources(Op) >= 2)
      y = goc::fp64_input(b, lane, mode >> 1);
    double value;
    if constexpr (Op == goc::Fp64::Add)
      value = x + y;
    if constexpr (Op == goc::Fp64::Mul)
      value = x * y;
    if constexpr (Op == goc::Fp64::Fma)
      value = std::fma(x, y, goc::fp64_input(c, lane, mode >> 2));
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
    uint64_t bits = goc::double_bits(goc::fp64_output(value, mode));
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

} // namespace

int goc_rdna4_v_add_f64(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b) {
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic<goc::Fp64::Add>(flags, exec_mask, mode, d, a, b, nullptr);
}

int goc_rdna4_v_mul_f64(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b) {
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic<goc::Fp64::Mul>(flags, exec_mask, mode, d, a, b, nullptr);
}

int goc_rdna4_v_fma_f64(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c) {
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic<goc::Fp64::Fma>(flags, exec_mask, mode, d, a, b, c);
}

int goc_rdna4_v_trunc_f64(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                          const uint32_t *const *a) {
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic<goc::Fp64::Trunc>(flags, exec_mask, mode, d, a, nullptr, nullptr);
}

int goc_rdna4_v_ceil_f64(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a) {
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic<goc::Fp64::Ceil>(flags, exec_mask, mode, d, a, nullptr, nullptr);
}

int goc_rdna4_v_rndne_f64(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                          const uint32_t *const *a) {
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic<goc::Fp64::Rndne>(flags, exec_mask, mode, d, a, nullptr, nullptr);
}

int goc_rdna4_v_floor_f64(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                          const uint32_t *const *a) {
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic<goc::Fp64::Floor>(flags, exec_mask, mode, d, a, nullptr, nullptr);
}

int goc_rdna4_v_fract_f64(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                          const uint32_t *const *a) {
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic<goc::Fp64::Fract>(flags, exec_mask, mode, d, a, nullptr, nullptr);
}

int goc_rdna4_v_sqrt_f64(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a) {
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic<goc::Fp64::Sqrt>(flags, exec_mask, mode, d, a, nullptr, nullptr);
}

int goc_rdna4_v_rcp_f64(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                        const uint32_t *const *a) {
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic<goc::Fp64::Rcp>(flags, exec_mask, mode, d, a, nullptr, nullptr);
}

int goc_rdna4_v_rsq_f64(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                        const uint32_t *const *a) {
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic<goc::Fp64::Rsq>(flags, exec_mask, mode, d, a, nullptr, nullptr);
}

int goc_rdna4_v_min_num_f64(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b) {
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic<goc::Fp64::MinNum>(flags, exec_mask, mode, d, a, b, nullptr);
}

int goc_rdna4_v_max_num_f64(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b) {
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic<goc::Fp64::MaxNum>(flags, exec_mask, mode, d, a, b, nullptr);
}

int goc_rdna4_v_minimum_f64(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b) {
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic<goc::Fp64::Minimum>(flags, exec_mask, mode, d, a, b, nullptr);
}

int goc_rdna4_v_maximum_f64(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b) {
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic<goc::Fp64::Maximum>(flags, exec_mask, mode, d, a, b, nullptr);
}

int goc_rdna4_v_frexp_mant_f64(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a) {
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return arithmetic<goc::Fp64::FrexpMant>(flags, exec_mask, mode, d, a, nullptr, nullptr);
}

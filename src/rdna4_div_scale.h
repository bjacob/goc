// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Pre-scaling decisions adapted from rocjitsu shared/division.h.

#pragma once

#include "goc/goc.h"
#include "rdna4_division.h"

#include <stdint.h>

namespace goc {

template <unsigned Width> struct DivisionScaleResult {
  typename DivisionFormat<Width>::Bits value;
  bool post_scale;
};

// Pre-scale A (equal to B or C after NEG modifiers), with B the denominator
// and C the numerator. Returns the scaled value and post-scaling condition.
// Uses nearest-even rounding with denormals enabled and preserves host FP state.
template <unsigned Width>
inline DivisionScaleResult<Width>
division_scale(typename DivisionFormat<Width>::Bits a, typename DivisionFormat<Width>::Bits b,
               typename DivisionFormat<Width>::Bits c, uint32_t mode) {
  using F = DivisionFormat<Width>;
  using T = typename F::Bits;
  constexpr int threshold = Width == 32 ? 96 : 768, scale = Width == 32 ? 64 : 128;
  if (mode & GOC_ALU_NEG_A)
    a ^= F::sign;
  if (mode & GOC_ALU_NEG_B)
    b ^= F::sign;
  if (mode & GOC_ALU_NEG_C)
    c ^= F::sign;
  T bm = b & (F::sign - 1), cm = c & (F::sign - 1);
  int de = int((b & F::infinity) >> F::fraction), ne = int((c & F::infinity) >> F::fraction),
      delta = ne - de;
  int adjustment = 0;
  bool post_scale = false, is_denominator = a == b;
  if (delta >= threshold) {
    adjustment = is_denominator ? scale : 0;
    post_scale = true;
  } else if (de == 0)
    adjustment = scale;
  else if (de >= 2 * F::bias - 1) {
    post_scale = delta <= -threshold;
    adjustment = post_scale && !is_denominator ? 0 : -scale;
  } else if (delta <= -threshold) {
    adjustment = is_denominator ? 0 : scale;
    post_scale = true;
  } else if (ne <= F::fraction + 1)
    adjustment = scale;
  T value;
  if (!bm || !cm)
    value = F::sign | F::infinity | F::quiet;
  else
    value = division_scale_bits<Width>(a, adjustment);
  return {division_output<Width>(value, mode), post_scale};
}

template <unsigned Width>
uint32_t div_scale_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c);

template <unsigned Width>
uint32_t div_scale_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c);

} // namespace goc

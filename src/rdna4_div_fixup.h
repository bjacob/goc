// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Adapted from rocjitsu shared/division.h; FP16 saturation and OMOD behavior
// follow RX 9070 captures rather than the promoted-FP32 rocjitsu implementation.

#pragma once

#include "goc/goc.h"

#include <stdint.h>
#include <type_traits>

namespace goc {

template <unsigned Width> struct FixupFormat {
  using Bits = std::conditional_t<Width == 64, uint64_t, uint32_t>;
  static constexpr int fraction = Width == 16 ? 10 : Width == 32 ? 23 : 52;
  static constexpr int bias = Width == 16 ? 15 : Width == 32 ? 127 : 1023;
  static constexpr Bits sign = Bits(1) << (Width - 1);
  static constexpr Bits infinity = Bits(2 * bias + 1) << fraction;
  static constexpr Bits quiet = Bits(1) << (fraction - 1);
};

// Repair provisional quotient P using denominator B and numerator C, preserving
// denormals and applying nearest-even overflow rules. Applies all source/output
// modifiers and FP16 saturation without reading or changing host FP state.
template <unsigned Width>
inline typename FixupFormat<Width>::Bits
fixup_value(typename FixupFormat<Width>::Bits p, typename FixupFormat<Width>::Bits b,
            typename FixupFormat<Width>::Bits c, uint32_t mode, bool saturate) {
  using F = FixupFormat<Width>;
  using T = typename F::Bits;
  T inputs[] = {p, b, c};
  for (unsigned i = 0; i < 3; ++i) {
    if (mode & (GOC_ALU_ABS_A << i))
      inputs[i] &= F::sign - 1;
    if (mode & (GOC_ALU_NEG_A << i))
      inputs[i] ^= F::sign;
  }
  p = inputs[0];
  b = inputs[1];
  c = inputs[2];
  T bm = b & (F::sign - 1), cm = c & (F::sign - 1), pm = p & (F::sign - 1),
    sign = (b ^ c) & F::sign, result;
  T overflow = F::infinity - (Width == 16 && saturate);
  if (cm > F::infinity)
    result = c | F::quiet;
  else if (bm > F::infinity)
    result = b | F::quiet;
  else if ((bm == 0 && cm == 0) || (bm == F::infinity && cm == F::infinity))
    result = F::sign | F::infinity | F::quiet;
  else if (bm == 0 || cm == F::infinity)
    result = sign | F::infinity;
  else if (cm == 0 || bm == F::infinity)
    result = sign;
  else if (Width != 16 &&
           int(cm >> F::fraction) - int(bm >> F::fraction) < -(F::bias + F::fraction))
    result = sign;
  else
    result = sign | (pm >= F::infinity ? overflow : pm);
  unsigned omod = (mode >> 6) & 3;
  if (omod) {
    T magnitude = result & (F::sign - 1), out_sign = result & F::sign;
    int exponent = int((result & F::infinity) >> F::fraction);
    if (!exponent)
      result = 0;
    else if (magnitude < F::infinity) {
      exponent += omod == 3 ? -1 : int(omod);
      if (exponent <= 0)
        result = out_sign;
      else if (exponent >= 2 * F::bias + 1)
        result = out_sign | overflow;
      else
        result = (result & ~F::infinity) | (T(exponent) << F::fraction);
    }
  }
  if (mode & GOC_ALU_CLAMP) {
    if ((result & F::sign) || (result > F::infinity))
      result = 0;
    else if (result > (T(F::bias) << F::fraction))
      result = T(F::bias) << F::fraction;
  }
  return result;
}

template <unsigned Width>
void fixup_x86_64_v3(uint32_t mask, uint32_t mode, bool saturate, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c);

template <unsigned Width>
void fixup_x86_64_v4(uint32_t mask, uint32_t mode, bool saturate, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c);

} // namespace goc

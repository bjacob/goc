// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Format and rounding helpers adapted from rocjitsu shared/division.h.

#pragma once

#include "goc/goc.h"

#include <stdint.h>
#include <type_traits>

namespace goc {

template <unsigned Width> struct DivisionFormat {
  using Bits = std::conditional_t<Width == 64, uint64_t, uint32_t>;
  static constexpr int fraction = Width == 16 ? 10 : Width == 32 ? 23 : 52;
  static constexpr int bias = Width == 16 ? 15 : Width == 32 ? 127 : 1023;
  static constexpr Bits sign = Bits(1) << (Width - 1);
  static constexpr Bits infinity = Bits(2 * bias + 1) << fraction;
  static constexpr Bits quiet = Bits(1) << (fraction - 1);
};

// Scale IEEE bits by a power of two with nearest-even rounding and gradual
// underflow. Preserve zeros and infinities, and quiet NaNs without losing payload.
template <unsigned Width>
inline typename DivisionFormat<Width>::Bits
division_scale_bits(typename DivisionFormat<Width>::Bits raw, int adjustment) {
  using F = DivisionFormat<Width>;
  using T = typename F::Bits;
  T sign = raw & F::sign, magnitude = raw & (F::sign - 1),
    fraction_mask = (T(1) << F::fraction) - 1;
  if (magnitude >= F::infinity)
    return raw | (magnitude > F::infinity ? F::quiet : 0);
  if (!magnitude)
    return raw;
  int exponent = int(magnitude >> F::fraction);
  T significand = magnitude & fraction_mask;
  if (exponent)
    significand |= T(1) << F::fraction;
  else {
    exponent = 1;
    while (significand < (T(1) << F::fraction)) {
      significand <<= 1;
      --exponent;
    }
  }
  exponent += adjustment;
  if (exponent >= 2 * F::bias + 1)
    return sign | F::infinity;
  if (exponent > 0)
    return sign | (T(exponent) << F::fraction) | (significand & fraction_mask);
  unsigned shift = unsigned(1 - exponent);
  if (shift > F::fraction + 1)
    return sign;
  T rounded = (significand + (T(1) << (shift - 1)) - 1 + ((significand >> shift) & 1)) >> shift;
  return sign | rounded;
}

// Apply OMOD/CLAMP to IEEE bits. Active OMOD flushes input subnormals to +0
// and output subnormals to signed zero. Finite overflow uses the supplied code.
template <unsigned Width>
inline typename DivisionFormat<Width>::Bits
division_output(typename DivisionFormat<Width>::Bits result, uint32_t mode,
                typename DivisionFormat<Width>::Bits overflow = DivisionFormat<Width>::infinity) {
  using F = DivisionFormat<Width>;
  using T = typename F::Bits;
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

} // namespace goc

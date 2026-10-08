// SPDX-License-Identifier: MIT
// rndne is adapted from rocjitsu util::rndne_scalar (MIT).

#pragma once

#include "goc/goc.h"
#include "internal.h"

#include <cmath>
#include <stdint.h>

namespace goc {

enum class Unary { Trunc, Ceil, Rndne, Floor, Sqrt, Rcp, Rsq, Exp, Log, Fract, FrexpMant };

// Round to an integral FP32 value, ties to even, preserving signed zero and
// quieting NaNs without depending on the host rounding mode.
inline float rndne(float value) {
  const uint32_t bits = as_bits(value), magnitude = bits & 0x7fffffff, sign = bits & 0x80000000;
  if (magnitude >= 0x7f800000)
    return as_float(bits | (magnitude > 0x7f800000 ? 0x00400000 : 0));
  const unsigned exponent = magnitude >> 23;
  if (exponent >= 150)
    return value;
  if (exponent < 127)
    return as_float(sign | (magnitude > 0x3f000000 ? 0x3f800000 : 0));
  const uint32_t unit = UINT32_C(1) << (150 - exponent);
  const uint32_t fraction = magnitude & (unit - 1);
  uint32_t rounded = magnitude & ~(unit - 1);
  if (fraction > unit / 2 || (fraction == unit / 2 && (rounded & unit)))
    rounded += unit;
  return as_float(sign | rounded);
}

template <Unary Op> inline float unary_value(float value) {
  if constexpr (Op == Unary::FrexpMant) {
    uint32_t magnitude = as_bits(value) & 0x7fffffff;
    if (magnitude == 0 || magnitude >= 0x7f800000)
      return value;
    int exponent;
    return std::frexp(value, &exponent);
  }
  if constexpr (Op == Unary::Trunc)
    return std::trunc(value);
  if constexpr (Op == Unary::Ceil)
    return std::ceil(value);
  if constexpr (Op == Unary::Rndne)
    return rndne(value);
  if constexpr (Op == Unary::Floor)
    return std::floor(value);
  if constexpr (Op == Unary::Sqrt)
    return std::sqrt(value);
  if constexpr (Op == Unary::Rcp)
    return 1.0f / value;
  if constexpr (Op == Unary::Rsq)
    return 1.0f / std::sqrt(value);
  if constexpr (Op == Unary::Exp)
    return std::exp2(value);
  if constexpr (Op == Unary::Fract) {
    float result = value - std::floor(value);
    return result > as_float(0x3f7fffff) ? as_float(0x3f7fffff) : result;
  }
  if constexpr (Op == Unary::Log)
    return std::log2(value);
}

#if defined(GOC_HAVE_X86_64_V3)
void unary_x86_64_v3(Unary op, uint32_t mask, uint32_t modifiers, uint32_t *d, const uint32_t *a);
#endif

} // namespace goc

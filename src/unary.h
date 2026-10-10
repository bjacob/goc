// SPDX-License-Identifier: MIT
// rndne is adapted from rocjitsu util::rndne_scalar (MIT).

#pragma once

#include "goc/goc.h"
#include "internal.h"
#include "round_integral.h"

#include <cmath>
#include <stdint.h>

namespace goc {

enum class Unary { Trunc, Ceil, Rndne, Floor, Sqrt, Rcp, Rsq, Exp, Log, Fract, FrexpMant };

template <Unary Op>
inline constexpr bool unary_flushes_f32 = Op == Unary::Sqrt || Op == Unary::Rcp ||
                                          Op == Unary::Rsq || Op == Unary::Exp || Op == Unary::Log;

template <Unary Op>
inline constexpr bool unary_rounds_f32 =
    Op == Unary::Trunc || Op == Unary::Ceil || Op == Unary::Rndne || Op == Unary::Floor;

// Round to an integral FP32 value, ties to even, preserving signed zero and
// quieting NaNs without depending on the host rounding mode.
inline float rndne(float value) {
  return as_float(round_integral_bits<IntegralRound::Nearest, false>(as_bits(value), false));
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
void unary_x86_64_v3(Unary op, uint32_t exec_mask, uint32_t modifiers, uint32_t *d,
                     const uint32_t *a, bool flush_input);
#endif

} // namespace goc

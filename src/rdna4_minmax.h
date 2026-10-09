// SPDX-License-Identifier: MIT

#pragma once

#include "float_bits.h"
#include "internal.h"

#include <stdint.h>

namespace goc {

// IEEE selection orders -0 below +0. Number variants ignore even signaling NaNs
// when the other operand is numeric; propagating variants prefer signaling NaNs.
template <bool Maximum, bool Propagate, typename Float> Float minmax(Float x, Float y) {
  using Bits = FloatBits<Float>;
  typename Bits::UInt a = Bits::bits(x), b = Bits::bits(y);
  bool an = (a & Bits::magnitude) > Bits::infinity, bn = (b & Bits::magnitude) > Bits::infinity;
  if constexpr (Propagate) {
    if (an && !(a & Bits::quiet))
      return Bits::value(a | Bits::quiet);
    if (bn && !(b & Bits::quiet))
      return Bits::value(b | Bits::quiet);
    if (an || bn)
      return Bits::value((an ? a : b) | Bits::quiet);
  } else {
    if (an)
      return bn ? Bits::value(a | Bits::quiet) : y;
    if (bn)
      return x;
  }
  if (x == y)
    return Bits::value(Maximum ? (a & b) : (a | b));
  return (Maximum ? x > y : x < y) ? x : y;
}

// Select A/B first and then C; median follows the ISA's first-maximum removal
// rule by default. OrderedMedian orders -0 below +0 for FP16. Either median
// returns minimumNumber(A,B,C) if any input is NaN.
template <bool FirstMaximum, bool SecondMaximum, bool Propagate, bool Median = false,
          bool OrderedMedian = false>
float minmax3_value(float x, float y, float z) {
  float value;
  if constexpr (Median) {
    const bool has_nan = (goc::as_bits(x) & 0x7fffffff) > 0x7f800000 ||
                         (goc::as_bits(y) & 0x7fffffff) > 0x7f800000 ||
                         (goc::as_bits(z) & 0x7fffffff) > 0x7f800000;
    if (has_nan) {
      value = goc::minmax<false, false>(goc::minmax<false, false>(x, y), z);
    } else if constexpr (OrderedMedian) {
      value =
          goc::minmax<true, false>(goc::minmax<false, false>(x, y),
                                   goc::minmax<false, false>(goc::minmax<true, false>(x, y), z));
    } else {
      float maximum = goc::minmax<true, false>(goc::minmax<true, false>(x, y), z);
      value = maximum == x   ? goc::minmax<true, false>(y, z)
              : maximum == y ? goc::minmax<true, false>(x, z)
                             : goc::minmax<true, false>(x, y);
    }
  } else {
    float ab = goc::minmax<FirstMaximum, Propagate>(x, y);
    value = goc::minmax<SecondMaximum, Propagate>(ab, z);
  }
  return value;
}

enum class Minmax3 {
  MedianNum,
  Min3Num,
  Max3Num,
  MinmaxNum,
  MaxminNum,
  Minimum3,
  Maximum3,
  MinimumMaximum,
  MaximumMinimum,
};

#if defined(GOC_HAVE_X86_64_V3)
void minmax3_x86_64_v3(Minmax3 op, uint32_t mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                       const uint32_t *b, const uint32_t *c);
#endif

} // namespace goc

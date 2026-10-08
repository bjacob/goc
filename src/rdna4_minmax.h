// SPDX-License-Identifier: MIT

#pragma once

#include "internal.h"

#include <stdint.h>

namespace goc {

// IEEE selection orders -0 below +0. Number variants ignore even signaling NaNs
// when the other operand is numeric; propagating variants prefer signaling NaNs.
template <bool Maximum, bool Propagate> float minmax(float x, float y) {
  uint32_t a = goc::as_bits(x), b = goc::as_bits(y);
  bool an = (a & 0x7fffffff) > 0x7f800000, bn = (b & 0x7fffffff) > 0x7f800000;
  if constexpr (Propagate) {
    if (an && !(a & 0x00400000))
      return goc::as_float(a | 0x00400000);
    if (bn && !(b & 0x00400000))
      return goc::as_float(b | 0x00400000);
    if (an || bn)
      return goc::as_float((an ? a : b) | 0x00400000);
  } else {
    if (an)
      return bn ? goc::as_float(a | 0x00400000) : y;
    if (bn)
      return x;
  }
  if (x == y)
    return goc::as_float(Maximum ? (a & b) : (a | b));
  return (Maximum ? x > y : x < y) ? x : y;
}

enum class Minmax3 {
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

// SPDX-License-Identifier: MIT

#pragma once

#include "fma_integer.h"
#include "goc/goc.h"

#include <stdint.h>

namespace goc {

// rocjitsu shared/fp_mode.h FMA NaN ordering and nearest-even rounding,
// using the existing integer intermediate instead of host FP-control changes.
// Inputs have already been decoded, modified and flushed by their source type.
inline uint32_t fma_value_f32(uint32_t a, uint32_t b, uint32_t c, bool clamp, uint64_t flags,
                              bool rdna3) {
  uint32_t ma = a & 0x7fffffff, mb = b & 0x7fffffff, mc = c & 0x7fffffff;

  uint32_t result;
  if ((ma == 0 && mb == 0x7f800000) || (mb == 0 && ma == 0x7f800000))
    result = 0xffc00000;
  else if (ma > 0x7f800000 || mb > 0x7f800000 || mc > 0x7f800000)
    result = (ma > 0x7f800000   ? a
              : mb > 0x7f800000 ? b
                                : c) |
             (!rdna3 || (flags & GOC_FP_IEEE_MODE) ? 0x00400000 : 0);
  else if (ma == 0x7f800000 || mb == 0x7f800000) {
    result = ((a ^ b) & 0x80000000) | 0x7f800000;
    if (mc == 0x7f800000 && ((result ^ c) & 0x80000000))
      result = 0xffc00000;
  } else if (mc == 0x7f800000)
    result = c;
  else {
    uint64_t value = fma_finite_round_odd(a, b, c);
    result = fma_narrow_f32(value, flags & GOC_FP_FLUSH_OUTPUT_DENORMALS);
  }
  if (clamp)
    result = (result & 0x80000000) || (result & 0x7fffffff) > 0x7f800000 ? 0
             : result > 0x3f800000                                       ? 0x3f800000
                                                                         : result;
  return result;
}

} // namespace goc

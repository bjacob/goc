// SPDX-License-Identifier: MIT

#pragma once

#include "fma_integer.h"
#include "goc/goc.h"

#include <stdint.h>

namespace goc {

// Architectural FMA NaN ordering followed by integer nearest-even rounding.
// Inputs have already been decoded, modified and flushed by their source type.
// A non-null exceptions pointer accumulates RDNA4 FMA exception bits.
inline uint32_t fma_value_f32(uint32_t a, uint32_t b, uint32_t c, bool clamp, uint64_t flags,
                              bool rdna3, uint32_t *exceptions = nullptr) {
  uint32_t ma = a & 0x7fffffff, mb = b & 0x7fffffff, mc = c & 0x7fffffff;
  if (exceptions && !clamp) {
    bool nan = ma > 0x7f800000 || mb > 0x7f800000 || mc > 0x7f800000;
    bool invalid = (ma == 0 && mb == 0x7f800000) || (mb == 0 && ma == 0x7f800000) ||
                   (ma > 0x7f800000 && !(ma & 0x400000)) || (mb > 0x7f800000 && !(mb & 0x400000)) ||
                   (mc > 0x7f800000 && !(mc & 0x400000)) ||
                   (!nan && (ma == 0x7f800000 || mb == 0x7f800000) && mc == 0x7f800000 &&
                    ((a ^ b ^ c) & 0x80000000));
    if (invalid)
      *exceptions |= GOC_EXCEPTION_INVALID;
    else if (!nan && ((ma && ma < 0x800000) || (mb && mb < 0x800000) || (mc && mc < 0x800000)))
      *exceptions |= GOC_EXCEPTION_INPUT_DENORM;
  }
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
    if (exceptions && !clamp) {
      uint64_t magnitude = value & 0x7fffffffffffffffULL;
      if ((result & 0x7fffffff) == 0x7f800000)
        *exceptions |= GOC_EXCEPTION_OVERFLOW | GOC_EXCEPTION_INEXACT;
      else if (magnitude) {
        int exponent = int(magnitude >> 52) - 1023;
        unsigned shift = 29 + (exponent < -126 ? -126 - exponent : 0);
        // Round-to-odd retains all information needed to detect discarded bits.
        bool inexact = !(result & 0x7fffffff) || shift >= 64 ||
                       ((magnitude & 0xfffffffffffffULL) | (1ULL << 52)) & ((1ULL << shift) - 1);
        if (inexact) {
          *exceptions |= GOC_EXCEPTION_INEXACT;
          // 2^-126 - 2^-151: tininess after significand rounding, before packing.
          if (magnitude < 0x380ffffff0000000ULL)
            *exceptions |= GOC_EXCEPTION_UNDERFLOW;
        }
      }
    }
  }
  if (clamp)
    result = (result & 0x80000000) || (result & 0x7fffffff) > 0x7f800000 ? 0
             : result > 0x3f800000                                       ? 0x3f800000
                                                                         : result;
  return result;
}

} // namespace goc

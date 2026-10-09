// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"
#include "rdna4_conversion64_reference.h"

#include <stdint.h>

namespace goc_test {

inline unsigned packed_conversion_modes(unsigned op) { return op == 0 ? 128 : 32; }

inline uint32_t packed_conversion_mode(unsigned op, unsigned variant) {
  uint32_t mode = (variant & 1 ? GOC_ALU_NEG_A : 0) | (variant & 2 ? GOC_ALU_NEG_B : 0) |
                  (variant & 4 ? GOC_ALU_ABS_A : 0) | (variant & 8 ? GOC_ALU_ABS_B : 0) |
                  (variant & 16 ? GOC_ALU_CLAMP : 0);
  return mode | (op == 0 ? (variant >> 5) << 6 : 0);
}

inline uint16_t packed_conversion_half_reference(unsigned op, uint32_t raw, uint32_t mode) {
  if (mode & GOC_ALU_ABS_A)
    raw &= 0x7fffffff;
  if (mode & GOC_ALU_NEG_A)
    raw ^= 0x80000000;
  unsigned magnitude = raw & 0x7fffffff;
  bool negative = raw >> 31;
  if (op == 0) {
    uint16_t sign = negative ? 0x8000 : 0;
    if (magnitude >= 0x7f800000)
      return uint16_t(sign | 0x7c00 |
                      (magnitude > 0x7f800000 ? ((raw & 0x7fffff) >> 13) | 0x200 : 0));
    // Find the greatest positive half no larger than the input. FP32 bit
    // ordering and integer-only widening make this independent of host FP.
    unsigned low = 0, high = 0x7bff;
    while (low < high) {
      unsigned middle = (low + high + 1) / 2;
      if (conversion_reencode(middle, 10, 15, 23, 127) <= magnitude)
        low = middle;
      else
        high = middle - 1;
    }
    return uint16_t(sign | low);
  }
  if (magnitude > 0x7f800000 || magnitude < 0x3f800000 || (op == 2 && negative))
    return 0;
  uint32_t limit = op == 2 ? 65535 : negative ? 32768 : 32767;
  unsigned exponent = magnitude >> 23;
  uint32_t integer = exponent >= 143 ? limit : (0x800000 | (raw & 0x7fffff)) >> (150 - exponent);
  if (integer > limit)
    integer = limit;
  return uint16_t(negative ? 0u - integer : integer);
}

inline uint32_t packed_conversion_reference(unsigned op, uint32_t a, uint32_t b, uint32_t mode) {
  return packed_conversion_half_reference(op, a, mode) |
         (uint32_t(packed_conversion_half_reference(op, b, mode >> 1)) << 16);
}

} // namespace goc_test

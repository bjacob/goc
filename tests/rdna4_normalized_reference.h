// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"
#include "rdna4_conversion64_reference.h"

#include <stdint.h>

namespace goc_test {

inline unsigned normalized_modes(unsigned op) { return op < 2 ? 32 : 128; }

inline uint32_t normalized_mode(unsigned op, unsigned variant) {
  const uint32_t bits[] = {GOC_ALU_NEG_A,
                           GOC_ALU_ABS_A,
                           GOC_ALU_CLAMP,
                           op < 4 ? GOC_ALU_NEG_B : 0,
                           op < 4 ? GOC_ALU_ABS_B : 0,
                           op >= 2 ? GOC_ALU_HIGH_A : 0,
                           op >= 2 ? (op < 4 ? GOC_ALU_HIGH_B : GOC_ALU_HIGH_D) : 0,
                           op >= 4 ? GOC_ALU_OMOD_2 : 0,
                           op >= 4 ? GOC_ALU_OMOD_4 : 0};
  uint32_t mode = 0;
  for (uint32_t bit : bits)
    if (bit) {
      if (variant & 1)
        mode |= bit;
      variant >>= 1;
    }
  return mode;
}

inline uint16_t normalized_half_reference(unsigned op, uint32_t raw, uint32_t mode) {
  if (op >= 2)
    raw = uint32_t(
        conversion_reencode(uint16_t(raw >> (mode & GOC_ALU_HIGH_A ? 16 : 0)), 10, 15, 23, 127));
  if (mode & GOC_ALU_ABS_A)
    raw &= 0x7fffffff;
  if (mode & GOC_ALU_NEG_A)
    raw ^= 0x80000000;
  unsigned magnitude = raw & 0x7fffffff;
  bool negative = raw >> 31;
  if (magnitude > 0x7f800000 || ((op & 1) && negative))
    return 0;
  uint32_t scale = op & 1 ? 65535 : 32767;
  uint32_t result;
  if (magnitude >= 0x3f800000)
    result = scale;
  else {
    unsigned exponent = magnitude >> 23;
    uint64_t significand = (magnitude & 0x7fffff) | (exponent ? 0x800000 : 0);
    // Multiply the exact integer significand, then perform a single tie-even
    // right shift. No host FP multiplication or intermediate rounding.
    result =
        uint32_t(conversion_round_right(significand * scale, exponent ? 150 - int(exponent) : 149));
  }
  return uint16_t(negative ? 0u - result : result);
}

inline uint32_t normalized_reference(unsigned op, uint32_t a, uint32_t b, uint32_t original,
                                     uint32_t mode) {
  uint32_t low = normalized_half_reference(op, a, mode);
  if (op < 4)
    return low | (uint32_t(normalized_half_reference(op, b, mode >> 1)) << 16);
  unsigned shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  return (original & ~(65535u << shift)) | (low << shift);
}

} // namespace goc_test

// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <stdint.h>

namespace goc_test {

inline uint32_t byte_conversion_mode(unsigned variant) {
  return ((variant & 3) << 6) | (variant & 4 ? GOC_ALU_CLAMP : 0);
}

inline uint32_t byte_conversion_reference(unsigned byte, uint32_t raw, uint32_t mode) {
  // Encode twice the mathematical result as an integer; even div:2 is exact.
  const unsigned shifts[] = {1, 2, 3, 0};
  unsigned twice = ((raw >> (8 * byte)) & 255) << shifts[(mode >> 6) & 3];
  if ((mode & GOC_ALU_CLAMP) && twice > 2)
    twice = 2;
  if (!twice)
    return 0;
  unsigned exponent = 0;
  for (unsigned remaining = twice; remaining >>= 1;)
    ++exponent;
  return ((126 + exponent) << 23) | ((twice << (23 - exponent)) & 0x7fffff);
}

inline uint32_t nibble_offset_reference(uint32_t raw, uint32_t mode) {
  unsigned nibble = raw & 15;
  bool negative = nibble >= 8;
  unsigned magnitude = negative ? 16 - nibble : nibble;
  const int powers[] = {-4, -3, -2, -5};
  int power = powers[(mode >> 6) & 3];
  if (!magnitude || ((mode & GOC_ALU_CLAMP) && negative))
    return 0;
  if ((mode & GOC_ALU_CLAMP) && magnitude > (1u << -power))
    return 0x3f800000;
  unsigned top = 0;
  for (unsigned copy = magnitude; copy >>= 1;)
    ++top;
  return (negative ? 0x80000000u : 0) | (uint32_t(127 + int(top) + power) << 23) |
         ((magnitude << (23 - top)) & 0x7fffff);
}

inline uint32_t byte_pack_mode(unsigned variant) {
  return (variant & 1 ? GOC_ALU_ABS_A : 0) | (variant & 2 ? GOC_ALU_NEG_A : 0) |
         (variant & 4 ? GOC_ALU_CLAMP : 0);
}

inline uint32_t byte_pack_reference(uint32_t raw, uint32_t selection, uint32_t original,
                                    uint32_t mode) {
  if (mode & GOC_ALU_ABS_A)
    raw &= 0x7fffffff;
  if (mode & GOC_ALU_NEG_A)
    raw ^= 0x80000000;
  uint32_t magnitude = raw & 0x7fffffff, byte = 0;
  if (!(raw >> 31) && magnitude <= 0x7f800000) {
    if (magnitude >= 0x437f0000)
      byte = 255;
    else if (magnitude >= 0x3f000000) {
      unsigned shift = 150 - (magnitude >> 23);
      uint32_t significand = 0x800000 | (magnitude & 0x7fffff);
      byte = significand >> shift;
      uint32_t tail = significand & ((1u << shift) - 1), half = 1u << (shift - 1);
      byte += tail > half || (tail == half && (byte & 1));
    }
  }
  unsigned shift = (selection & 3) * 8;
  return (original & ~(255u << shift)) | (byte << shift);
}

} // namespace goc_test

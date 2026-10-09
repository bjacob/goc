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

} // namespace goc_test

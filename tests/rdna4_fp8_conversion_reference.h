// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"
#include "rdna4_conversion64_reference.h"

#include <stdint.h>

namespace goc_test {

inline uint32_t fp8_conversion_reference(bool bf8, uint8_t code) {
  bool negative = code >> 7;
  unsigned fraction_bits = bf8 ? 2 : 3;
  unsigned magnitude = code & 127;
  if (bf8 && magnitude == 124)
    return (negative ? 0x80000000u : 0) | 0x7f800000;
  if ((bf8 && magnitude > 124) || (!bf8 && magnitude == 127))
    return (negative ? 0x80000000u : 0) | 0x7fc00000;
  unsigned exponent = magnitude >> fraction_bits;
  unsigned significand = magnitude & ((1u << fraction_bits) - 1);
  if (exponent)
    significand += 1u << fraction_bits;
  int power = int(exponent ? exponent : 1) - (bf8 ? 15 : 7) - int(fraction_bits);
  return uint32_t(conversion_encode(negative, significand, power, 23, 127));
}

inline uint32_t fp8_conversion_mode(bool packed, unsigned selection) {
  return packed ? (selection ? GOC_ALU_HIGH_A : 0) : selection << 16;
}

} // namespace goc_test

// SPDX-License-Identifier: MIT

#ifndef GOC_FLOAT_FORMATS_H_
#define GOC_FLOAT_FORMATS_H_

#include "internal.h"

#include <stdint.h>

namespace goc {

inline float bf16_to_float(uint16_t bits) { return as_float(uint32_t(bits) << 16); }

inline float f16_to_float(uint16_t bits) {
  uint32_t sign = uint32_t(bits & 0x8000) << 16;
  int exponent = (bits >> 10) & 31;
  uint32_t fraction = bits & 1023;
  if (exponent == 31)
    return as_float(sign | 0x7f800000 | (fraction << 13));
  if (exponent == 0) {
    if (!fraction)
      return as_float(sign);
    exponent = 1;
    while (!(fraction & 1024)) {
      fraction <<= 1;
      --exponent;
    }
    fraction &= 1023;
  }
  return as_float(sign | (uint32_t(exponent + 112) << 23) | (fraction << 13));
}

// OCP FP8 E4M3FN and BF8 E5M2, adapted from rocjitsu util/data_types.h.
// E4M3FN's top exponent remains finite except codes 0x7f/0xff; E5M2 has
// infinities. Neither uses the FNUZ encoding. All finite values widen exactly.
template <bool Bf8> inline float fp8_to_float(uint8_t bits) {
  constexpr int fraction_bits = Bf8 ? 2 : 3;
  constexpr int bias = Bf8 ? 15 : 7;
  constexpr int max_exponent = Bf8 ? 31 : 15;
  uint32_t sign = uint32_t(bits & 0x80) << 24;
  int exponent = (bits & 0x7f) >> fraction_bits;
  uint32_t fraction = bits & ((1u << fraction_bits) - 1);
  if (exponent == max_exponent) {
    if ((Bf8 && fraction) || (!Bf8 && fraction == 7))
      return as_float(sign | 0x7fc00000);
    if constexpr (Bf8)
      return as_float(sign | 0x7f800000);
  }
  if (!exponent) {
    if (!fraction)
      return as_float(sign);
    exponent = 1;
    while (!(fraction & (1u << fraction_bits))) {
      fraction <<= 1;
      --exponent;
    }
    fraction &= (1u << fraction_bits) - 1;
  }
  return as_float(sign | (uint32_t(exponent + 127 - bias) << 23) |
                  (fraction << (23 - fraction_bits)));
}

} // namespace goc

#endif

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
} // namespace goc

#endif

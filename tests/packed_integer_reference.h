// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <stdint.h>

namespace goc_test {

inline uint32_t packed_integer_reference(int op, uint32_t a, uint32_t b, uint32_t mode) {
  uint32_t result = 0;
  for (int half = 0; half < 2; ++half) {
    bool high_a = half ? !(mode & GOC_PK_HI_A_LOW) : bool(mode & GOC_PK_LO_A_HIGH);
    bool high_b = half ? !(mode & GOC_PK_HI_B_LOW) : bool(mode & GOC_PK_LO_B_HIGH);
    int64_t x = (a >> (high_a ? 16 : 0)) & 65535;
    int64_t y = (b >> (high_b ? 16 : 0)) & 65535;
    bool sign = op == 0 || op == 1 || op == 4 || op == 5;
    if (sign) {
      if (x >= 32768)
        x -= 65536;
      if (y >= 32768)
        y -= 65536;
    }
    int64_t value = op < 4     ? ((op & 1) ? x - y : x + y)
                    : op == 8  ? x * y
                    : (op & 1) ? (x > y ? x : y)
                               : (x < y ? x : y);
    if (op >= 9) {
      if (op == 11 && y >= 32768)
        y -= 65536;
      const int64_t power = INT64_C(1) << (x % 16);
      // Arithmetic right shift rounds negative quotients down, unlike C++ division.
      value = op == 9 ? y * power : (y >= 0 ? y / power : -((-y + power - 1) / power));
    }
    if (op < 4 && (mode & GOC_PK_CLAMP)) {
      int64_t low = sign ? -32768 : 0, high = sign ? 32767 : 65535;
      if (value < low)
        value = low;
      if (value > high)
        value = high;
    }
    result |= (uint32_t(value) & 65535) << (16 * half);
  }
  return result;
}

} // namespace goc_test

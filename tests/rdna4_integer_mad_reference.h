// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <stdint.h>

namespace goc_test {

inline uint32_t integer_mad_reference(int op, uint32_t a, uint32_t b, uint32_t c, uint32_t mode) {
  const int bits = op < 2 ? 16 : 24;
  const int64_t range = INT64_C(1) << bits;
  int64_t x = (op < 2 && (mode & GOC_ALU_HIGH_A) ? a >> 16 : a) % range;
  int64_t y = (op < 2 && (mode & GOC_ALU_HIGH_B) ? b >> 16 : b) % range;
  int64_t z = c;
  if (op & 1) {
    if (x >= range / 2)
      x -= range;
    if (y >= range / 2)
      y -= range;
    if (z > INT32_MAX)
      z -= INT64_C(4294967296);
  }
  int64_t result = x * y + z;
  if (mode & GOC_ALU_CLAMP) {
    int64_t low = op & 1 ? INT32_MIN : 0, high = op & 1 ? INT32_MAX : int64_t(UINT32_MAX);
    if (result < low)
      result = low;
    if (result > high)
      result = high;
  }
  return uint32_t(result);
}

} // namespace goc_test

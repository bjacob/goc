// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <algorithm>
#include <cmath>
#include <stdint.h>

namespace goc_test {

inline float binary_reference(int op, float a, float b, uint32_t mode) {
  double x = a, y = b;
  if (mode & GOC_ALU_ABS_A)
    x = std::abs(x);
  if (mode & GOC_ALU_ABS_B)
    y = std::abs(y);
  if (mode & GOC_ALU_NEG_A)
    x = -x;
  if (mode & GOC_ALU_NEG_B)
    y = -y;
  float value = float(op == 0 ? x + y : op == 1 ? x - y : op == 2 ? y - x : x * y);
  if (op == 8 && (x == 0 || y == 0))
    value = 0;
  if (op >= 4 && op < 8) {
    bool maximum = op == 5 || op == 7;
    if (std::isnan(x) || std::isnan(y))
      value = op >= 6 ? NAN : std::isnan(x) ? y : x;
    else if (x == 0 && y == 0)
      value =
          (maximum ? (std::signbit(x) && std::signbit(y)) : (std::signbit(x) || std::signbit(y)))
              ? -0.0f
              : 0.0f;
    else
      value = maximum ? std::max(x, y) : std::min(x, y);
  }
  const float scales[] = {1, 2, 4, 0.5f};
  unsigned scale = (mode >> 6) & 3;
  if (scale && std::abs(value) < std::ldexp(1.0f, -126))
    value = 0;
  else if (scale == 3 && std::abs(value) < std::ldexp(1.0f, -125))
    value = std::copysign(0.0f, value);
  value *= scales[scale];
  if (mode & GOC_ALU_CLAMP)
    value = !(value > 0) ? 0 : std::min(value, 1.0f);
  return value;
}

} // namespace goc_test

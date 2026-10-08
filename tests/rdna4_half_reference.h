// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdint.h>

namespace goc_test {

// Numeric half reference independent of the implementation's bit conversions.
inline double half_value(uint16_t bits) {
  int exponent = (bits >> 10) & 31, fraction = bits & 1023;
  double magnitude = exponent == 31  ? (fraction ? std::numeric_limits<double>::quiet_NaN()
                                                 : std::numeric_limits<double>::infinity())
                     : exponent == 0 ? std::ldexp(double(fraction), -24)
                                     : std::ldexp(double(1024 + fraction), exponent - 25);
  return bits & 0x8000 ? -magnitude : magnitude;
}

inline uint16_t half_bits(double value, bool saturate = false) {
  uint16_t sign = std::signbit(value) ? 0x8000 : 0;
  if (std::isnan(value))
    return 0x7e00;
  if (std::isinf(value))
    return sign | 0x7c00;
  double magnitude = std::abs(value);
  if (magnitude >= 65520)
    return sign | (saturate ? 0x7bff : 0x7c00);
  if (magnitude == 0)
    return sign;
  int exponent;
  std::frexp(magnitude, &exponent);
  --exponent;
  double scaled = std::ldexp(magnitude, -std::max(exponent - 10, -24));
  unsigned rounded = unsigned(std::floor(scaled));
  double tail = scaled - rounded;
  rounded += tail > 0.5 || (tail == 0.5 && (rounded & 1));
  if (exponent < -14)
    return sign | uint16_t(rounded);
  return sign | uint16_t(((exponent + 14) << 10) + rounded);
}

// op: add, sub, subrev, mul, min_num, max_num, minimum, maximum.
inline double half_binary(int op, double x, double y) {
  switch (op) {
  case 0:
    return x + y;
  case 1:
    return x - y;
  case 2:
    return y - x;
  case 3:
    return x * y;
  default:
    if (std::isnan(x) || std::isnan(y)) {
      if (op >= 6)
        return std::numeric_limits<double>::quiet_NaN();
      return std::isnan(x) ? y : x;
    }
    bool maximum = op & 1;
    if (x == 0 && y == 0) {
      bool negative =
          maximum ? std::signbit(x) && std::signbit(y) : std::signbit(x) || std::signbit(y);
      return negative ? -0.0 : 0.0;
    }
    return maximum ? std::max(x, y) : std::min(x, y);
  }
}

} // namespace goc_test

// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <cmath>
#include <cstring>
#include <stdint.h>

namespace goc_test {

// Independent numeric reference: quantize in units of the output spacing.
inline uint32_t fp8_narrow_reference(bool bf8, bool stochastic, uint32_t raw, uint32_t seed,
                                     bool saturate) {
  unsigned sign = (raw >> 24) & 128, magnitude = raw & 0x7fffffff;
  unsigned terminal = bf8 ? 124 : 127;
  if (magnitude > 0x7f800000)
    return bf8 ? 0xfe : 0xff;
  if (magnitude == 0x7f800000)
    return sign | terminal;
  float input;
  std::memcpy(&input, &magnitude, sizeof(input));
  double x = input;
  unsigned limit = terminal - saturate;
  if (x > (bf8 ? 65536.0 : 512.0))
    return sign | limit;
  unsigned fraction_bits = bf8 ? 2 : 3;
  int exponent;
  std::frexp(x, &exponent);
  int power = exponent - 1 - int(fraction_bits);
  int minimum_power = bf8 ? -16 : -9;
  if (power < minimum_power || x == 0)
    power = minimum_power;
  double units = std::ldexp(x, -power);
  unsigned rounded;
  if (stochastic) {
    unsigned precision = 23 - fraction_bits;
    double fixed = std::floor(std::ldexp(units, int(precision)));
    rounded = unsigned(std::floor(std::ldexp(fixed + (seed >> (32 - precision)), -int(precision))));
  } else {
    rounded = unsigned(std::floor(units));
    double fraction = units - rounded;
    rounded += fraction > 0.5 || (fraction == 0.5 && (rounded & 1));
  }
  unsigned code = rounded + (unsigned(power - minimum_power) << fraction_bits);
  return sign | (code < limit ? code : limit);
}

inline uint32_t fp8_narrow_mode(unsigned op, unsigned variant) {
  uint32_t mode = (variant & 1 ? GOC_ALU_ABS_A : 0) | (variant & 2 ? GOC_ALU_NEG_A : 0);
  if (op >= 2)
    return mode | ((variant >> 2) << 16);
  return mode | (variant & 4 ? GOC_ALU_ABS_B : 0) | (variant & 8 ? GOC_ALU_NEG_B : 0) |
         (variant & 16 ? GOC_ALU_HIGH_D : 0);
}

inline uint32_t fp8_narrow_result(unsigned op, uint32_t a, uint32_t b, uint32_t old, uint32_t mode,
                                  bool saturate) {
  if (mode & GOC_ALU_ABS_A)
    a &= 0x7fffffff;
  if (mode & GOC_ALU_NEG_A)
    a ^= 0x80000000;
  uint32_t value = fp8_narrow_reference(op & 1, op >= 2, a, b, saturate);
  unsigned shift = ((mode >> 16) & 3) * 8;
  unsigned selected = 255;
  if (op < 2) {
    if (mode & GOC_ALU_ABS_B)
      b &= 0x7fffffff;
    if (mode & GOC_ALU_NEG_B)
      b ^= 0x80000000;
    value |= fp8_narrow_reference(op & 1, false, b, 0, saturate) << 8;
    shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
    selected = 65535;
  }
  return (old & ~(selected << shift)) | (value << shift);
}

} // namespace goc_test

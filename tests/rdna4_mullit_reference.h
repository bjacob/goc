// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <cmath>
#include <cstring>
#include <stdint.h>

namespace goc_test {

inline float mullit_float(uint32_t word) {
  float value;
  std::memcpy(&value, &word, sizeof(value));
  return value;
}

inline uint32_t mullit_bits(float value) {
  uint32_t word;
  std::memcpy(&word, &value, sizeof(word));
  return word;
}

// Evaluate the lighting rule with separate source and output modifiers.
inline uint32_t mullit_reference(uint32_t a, uint32_t b, uint32_t c, uint32_t mode) {
  const uint32_t words[] = {a, b, c};
  float input[3];
  for (unsigned i = 0; i < 3; ++i) {
    input[i] = mullit_float(words[i]);
    if (mode & (GOC_ALU_ABS_A << i))
      input[i] = std::fabs(input[i]);
    if (mode & (GOC_ALU_NEG_A << i))
      input[i] = -input[i];
  }
  float value;
  if (std::isnan(input[1]) || input[1] <= -mullit_float(0x7f7fffff) || std::isnan(input[2]) ||
      input[2] <= 0)
    value = -mullit_float(0x7f7fffff);
  else if (input[0] == 0 || input[1] == 0)
    value = 0;
  else
    value = input[0] * input[1];
  unsigned scale = (mode >> 6) & 3;
  if (scale) {
    if (std::fabs(value) < mullit_float(0x00800000))
      value = 0;
    value = std::ldexp(value, scale == 3 ? -1 : int(scale));
    if (std::fabs(value) < mullit_float(0x00800000))
      value = std::copysign(0.0f, value);
  }
  if (mode & GOC_ALU_CLAMP) {
    if (std::isnan(value) || value <= 0)
      value = 0;
    else if (value > 1)
      value = 1;
  }
  uint32_t word = mullit_bits(value);
  return (word & 0x7fffffff) > 0x7f800000 ? 0x7fc00000 : word;
}

} // namespace goc_test

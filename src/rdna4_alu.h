// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"
#include "internal.h"

#include <stdint.h>

namespace goc {

inline float alu_input(uint32_t bits, uint32_t modifiers) {
  if (modifiers & GOC_ALU_ABS_A)
    bits &= 0x7fffffff;
  if (modifiers & GOC_ALU_NEG_A)
    bits ^= 0x80000000;
  return as_float(bits);
}

inline float alu_output(float value, uint32_t modifiers) {
  switch ((modifiers >> 6) & 3) {
  case 1:
    value *= 2;
    break;
  case 2:
    value *= 4;
    break;
  case 3:
    value *= 0.5f;
    break;
  }
  if (modifiers & GOC_ALU_CLAMP)
    value = !(value > 0) ? 0 : (value > 1 ? 1 : value);
  return value;
}

} // namespace goc

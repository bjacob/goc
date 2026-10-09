// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"
#include "internal.h"

#include <stdint.h>

namespace goc {

// Replace FP32 subnormals with zero of the same sign.
inline float flush_denorm_f32(float value) {
  uint32_t raw = as_bits(value);
  return (raw & 0x7fffffff) < 0x00800000 ? as_float(raw & 0x80000000) : value;
}

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

// Scales an FP16 arithmetic result and applies CLAMP. OMOD flushes an
// unscaled tiny result to positive zero; halving a normal into the subnormal
// range produces signed zero instead. The input is represented in FP32.
inline float alu_output_f16(float value, uint32_t modifiers) {
  if (modifiers & GOC_ALU_OMOD_HALF) {
    uint32_t bits = as_bits(value), magnitude = bits & 0x7fffffffu;
    if (magnitude < 0x38800000u)
      value = 0;
    else if ((modifiers & GOC_ALU_OMOD_HALF) == GOC_ALU_OMOD_HALF && magnitude < 0x39000000u)
      value = as_float(bits & 0x80000000u);
  }
  return alu_output(value, modifiers);
}

// FP32 scaling follows rocjitsu's fp_mode::apply_omod_f32. Tiny unscaled
// results become +0; halving a normal result below twice minimum normal gives
// signed zero, independently of guest denormal mode.
inline float alu_output_f32(float value, uint32_t modifiers) {
  unsigned omod = (modifiers >> 6) & 3;
  if (omod) {
    uint32_t raw = as_bits(value), magnitude = raw & 0x7fffffff;
    if (magnitude < 0x00800000)
      value = 0.0f;
    else if (omod == 3 && magnitude < 0x01000000)
      value = as_float(raw & 0x80000000);
  }
  return alu_output(value, modifiers);
}

} // namespace goc

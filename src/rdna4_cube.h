// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Adapted from rocjitsu's shared/cube.h bit-level scalar and SIMD model.

#pragma once

#include "goc/goc.h"

#include <stdint.h>

namespace goc {

enum class Cube { Id, Sc, Tc, Ma };

// Magnitude used for cube comparisons, flushing subnormals to zero.
inline uint32_t cube_magnitude(uint32_t raw) {
  uint32_t magnitude = raw & 0x7fffffff;
  return magnitude < 0x00800000 ? 0 : magnitude;
}

// Execute cube arithmetic with source/output modifiers and nearest-even
// overflow handling. Quiet selected NaNs while retaining their sign/payload.
// Integer operations preserve host floating-point state.
template <Cube Op> inline uint32_t cube_value(uint32_t x, uint32_t y, uint32_t z, uint32_t mode) {
  x = (x & (mode & GOC_ALU_ABS_A ? 0x7fffffff : UINT32_MAX)) ^
      (mode & GOC_ALU_NEG_A ? 0x80000000 : 0);
  y = (y & (mode & GOC_ALU_ABS_B ? 0x7fffffff : UINT32_MAX)) ^
      (mode & GOC_ALU_NEG_B ? 0x80000000 : 0);
  z = (z & (mode & GOC_ALU_ABS_C ? 0x7fffffff : UINT32_MAX)) ^
      (mode & GOC_ALU_NEG_C ? 0x80000000 : 0);
  uint32_t ax = cube_magnitude(x), ay = cube_magnitude(y), az = cube_magnitude(z);
  bool z_axis = az <= 0x7f800000 && ax <= 0x7f800000 && ay <= 0x7f800000 && az >= ax && az >= ay;
  bool y_axis = ay <= 0x7f800000 && ax <= 0x7f800000 && ay >= ax;
  uint32_t major = z_axis ? z : y_axis ? y : x;
  uint32_t magnitude = cube_magnitude(major);
  bool negative = (major >> 31) && magnitude && magnitude <= 0x7f800000;
  uint32_t sign = negative ? 0x80000000 : 0, result;
  if constexpr (Op == Cube::Id)
    result = z_axis   ? (negative ? 0x40a00000 : 0x40800000)
             : y_axis ? (negative ? 0x40400000 : 0x40000000)
                      : (negative ? 0x3f800000 : 0);
  else if constexpr (Op == Cube::Sc)
    result = z_axis ? x ^ sign : y_axis ? x : z ^ sign ^ 0x80000000;
  else if constexpr (Op == Cube::Tc)
    result = !z_axis && y_axis ? z ^ sign : y ^ 0x80000000;
  else {
    result = magnitude >= 0x7f000000 ? (major & 0x80000000) | 0x7f800000 : major + 0x00800000;
    if (magnitude >= 0x7f800000)
      result = major;
    if (!magnitude)
      result = 0;
  }
  if ((result & 0x7fffffff) > 0x7f800000)
    result |= 0x00400000;
  unsigned omod = (mode >> 6) & 3;
  if (omod) {
    uint32_t exponent = result & 0x7f800000, output_sign = result & 0x80000000;
    if (!exponent)
      result = 0;
    else if (exponent != 0x7f800000) {
      if (omod == 3)
        result = exponent == 0x00800000 ? output_sign : result - 0x00800000;
      else
        result = exponent >= 0x7f800000 - omod * 0x00800000 ? output_sign | 0x7f800000
                                                            : result + omod * 0x00800000;
    }
  }
  if (mode & GOC_ALU_CLAMP) {
    if ((result >> 31) || result > 0x7f800000)
      result = 0;
    else if (result > 0x3f800000)
      result = 0x3f800000;
  }
  return result;
}

template <Cube Op>
void cube_x86_64_v3(uint32_t mask, uint32_t mode, uint32_t *d, const uint32_t *a, const uint32_t *b,
                    const uint32_t *c);

template <Cube Op>
void cube_x86_64_v4(uint32_t mask, uint32_t mode, uint32_t *d, const uint32_t *a, const uint32_t *b,
                    const uint32_t *c);

} // namespace goc

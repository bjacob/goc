// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "float_formats.h"
#include "fma_integer.h"
#include "fp64.h"
#include "goc/goc.h"
#include "half_fma_scalar.h"
#include "internal.h"

#include <stdint.h>

namespace goc {

// Decode one mixed source, with its flags shifted to the A positions.
inline uint32_t mixed_fma_input(uint32_t raw, uint32_t mode) {
  if (mode & GOC_MIX_F16_A)
    raw = as_bits(f16_to_float(uint16_t(raw >> (mode & GOC_ALU_HIGH_A ? 16 : 0))));
  if (mode & GOC_ALU_ABS_A)
    raw &= 0x7fffffff;
  if (mode & GOC_ALU_NEG_A)
    raw ^= 0x80000000;
  return raw;
}

// Return the FP16 FMA result when at least one FP32 input is nonfinite.
inline uint16_t mixed_fma_special(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t ma = a & 0x7fffffff, mb = b & 0x7fffffff, mc = c & 0x7fffffff;
  if ((ma == 0 && mb == 0x7f800000) || (mb == 0 && ma == 0x7f800000))
    return 0xfe00;
  if (ma > 0x7f800000 || mb > 0x7f800000 || mc > 0x7f800000) {
    uint32_t nan = ma > 0x7f800000 ? a : mb > 0x7f800000 ? b : c;
    return uint16_t(((nan >> 16) & 0x8000) | 0x7e00 | ((nan >> 13) & 1023));
  }
  if (ma == 0x7f800000 || mb == 0x7f800000) {
    uint32_t sign = (a ^ b) & 0x80000000;
    if (mc == 0x7f800000 && ((sign ^ c) & 0x80000000))
      return 0xfe00;
    return uint16_t((sign >> 16) | 0x7c00);
  }
  return uint16_t(((c >> 16) & 0x8000) | 0x7c00);
}

// Fuse FP32 inputs and round once to FP16, independently of host FP controls.
inline uint16_t mixed_fma_half_value(uint32_t a, uint32_t b, uint32_t c, bool clamp,
                                     bool saturate) {
  uint16_t result;
  if ((a & 0x7fffffff) >= 0x7f800000 || (b & 0x7fffffff) >= 0x7f800000 ||
      (c & 0x7fffffff) >= 0x7f800000) {
    result = mixed_fma_special(a, b, c);
  } else {
    result = half_fma_narrow(as_double(fma_finite_round_odd(a, b, c)), saturate);
  }
  return clamp ? half_fma_clamp(result) : result;
}

} // namespace goc

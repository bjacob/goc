// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "float_formats.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_fp64.h"
#include "rdna4_half_fma_scalar.h"

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

// Widen finite FP32 encodings without depending on host input-denormal controls.
inline double mixed_fma_widen(uint32_t bits) {
  uint64_t sign = uint64_t(bits & 0x80000000) << 32;
  int exponent = int((bits >> 23) & 255);
  uint32_t fraction = bits & 0x7fffff;
  if (!exponent) {
    if (!fraction)
      return as_double(sign);
    exponent = 1;
    while (!(fraction & 0x800000)) {
      fraction <<= 1;
      --exponent;
    }
    fraction &= 0x7fffff;
  }
  return as_double(sign | (uint64_t(exponent + 896) << 52) | (uint64_t(fraction) << 29));
}

// Round finite FP64 to FP16, nearest-even, saturating finite overflow if requested.
inline uint16_t mixed_fma_narrow(double value, bool saturate) {
  // Adapted from rocjitsu mixed_fma_simd.h: round_finite_f16_simd.
  uint64_t bits = double_bits(value), exponent = (bits >> 52) & 2047;
  uint16_t sign = uint16_t((bits >> 48) & 0x8000);
  if (exponent < 998)
    return sign;
  if (exponent > 1038)
    return sign | (saturate ? 0x7bff : 0x7c00);
  unsigned shift = exponent < 1009 ? unsigned(1051 - exponent) : 42;
  uint64_t significand = (bits & UINT64_C(0x000fffffffffffff)) | (UINT64_C(1) << 52);
  uint64_t rounded = significand >> shift;
  if (exponent >= 1009)
    rounded += (exponent - 1009) << 10;
  uint64_t remainder = significand & ((UINT64_C(1) << shift) - 1);
  uint64_t halfway = UINT64_C(1) << (shift - 1);
  rounded += remainder > halfway || (remainder == halfway && (rounded & 1));
  if (rounded >= 0x7c00)
    rounded = saturate ? 0x7bff : 0x7c00;
  return sign | uint16_t(rounded);
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

// Fuse FP32 inputs and round once to FP16 under nearest-even host arithmetic.
inline uint16_t mixed_fma_half_value(uint32_t a, uint32_t b, uint32_t c, bool clamp,
                                     bool saturate) {
  uint16_t result;
  if ((a & 0x7fffffff) >= 0x7f800000 || (b & 0x7fffffff) >= 0x7f800000 ||
      (c & 0x7fffffff) >= 0x7f800000) {
    result = mixed_fma_special(a, b, c);
  } else {
    // Borrowed from rocjitsu fma_f32_to_f16_nearest_environment: the FP32
    // product is exact in FP64; TwoSum and round-to-odd retain the addend.
    double product = mixed_fma_widen(a) * mixed_fma_widen(b);
    double addend = mixed_fma_widen(c), value = product + addend;
    double virtual_c = value - product;
    double error = (product - (value - virtual_c)) + (addend - virtual_c);
    uint64_t bits = double_bits(value), error_bits = double_bits(error);
    if (error != 0 && !(bits & 1))
      bits += ((bits ^ error_bits) >> 63) ? UINT64_MAX : UINT64_C(1);
    if (!(bits & UINT64_C(0x7fffffffffffffff)))
      bits = uint64_t(((a ^ b) & c & 0x80000000) && !(c & 0x7fffffff)) << 63;
    result = mixed_fma_narrow(as_double(bits), saturate);
  }
  return clamp ? half_fma_clamp(result) : result;
}

} // namespace goc

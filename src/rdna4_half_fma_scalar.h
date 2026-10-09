// SPDX-License-Identifier: MIT

#pragma once

#include "float_formats.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_fma_integer.h"
#include "rdna4_fp64.h"

#include <cmath>
#include <stdint.h>

namespace goc {

// Round finite FP64 to FP16, nearest-even, saturating finite overflow if requested.
inline uint16_t half_fma_narrow(double value, bool saturate) {
  // Adapted from rocjitsu mixed_fma_simd.h: round_finite_f16_simd.
  uint64_t bits = double_bits(value), exponent = (bits >> 52) & 2047;
  uint16_t sign = uint16_t((bits >> 48) & 0x8000);
  if (exponent < 998)
    return sign;
  if (exponent > 1038)
    return sign | (saturate ? 0x7bff : 0x7c00);
  unsigned shift = exponent < 1009 ? unsigned(1051 - exponent) : 42;
  uint64_t significand = (bits & 0x000fffffffffffffULL) | (1ULL << 52);
  uint64_t rounded = significand >> shift;
  if (exponent >= 1009)
    rounded += (exponent - 1009) << 10;
  uint64_t remainder = significand & ((1ULL << shift) - 1);
  uint64_t halfway = 1ULL << (shift - 1);
  rounded += remainder > halfway || (remainder == halfway && (rounded & 1));
  if (rounded >= 0x7c00)
    rounded = saturate ? 0x7bff : 0x7c00;
  return sign | uint16_t(rounded);
}

inline uint16_t half_fma_clamp(uint16_t value) {
  return (value & 0x8000) || (value & 0x7fff) > 0x7c00 ? 0 : value > 0x3c00 ? 0x3c00 : value;
}

// Adapted from rocjitsu shared/fp_mode.h: fma_f16 and finish_fma_f16.
// RNE and preserved input/output denormals are the currently exposed FP policy.
inline uint16_t half_fma_value(uint16_t a, uint16_t b, uint16_t c, uint32_t mode, bool saturate) {
  uint16_t inputs[] = {a, b, c};
  for (int i = 0; i < 3; ++i) {
    if (mode & (GOC_ALU_ABS_A << i))
      inputs[i] &= 0x7fff;
    if (mode & (GOC_ALU_NEG_A << i))
      inputs[i] ^= 0x8000;
  }
  a = inputs[0];
  b = inputs[1];
  c = inputs[2];
  uint16_t ma = a & 0x7fff, mb = b & 0x7fff, mc = c & 0x7fff;
  uint16_t exceptional = 0;
  if ((ma == 0 && mb == 0x7c00) || (mb == 0 && ma == 0x7c00))
    exceptional = 0xfe00;
  else if (ma > 0x7c00 || mb > 0x7c00 || mc > 0x7c00)
    exceptional = (ma > 0x7c00 ? a : mb > 0x7c00 ? b : c) | 0x0200;
  else if (ma == 0x7c00 || mb == 0x7c00) {
    exceptional = ((a ^ b) & 0x8000) | 0x7c00;
    if (mc == 0x7c00 && ((exceptional ^ c) & 0x8000))
      exceptional = 0xfe00;
  } else if (mc == 0x7c00)
    exceptional = c;
  if (exceptional)
    return mode & GOC_ALU_CLAMP ? half_fma_clamp(exceptional) : exceptional;

  double value = as_double(fma_finite_round_odd(as_bits(f16_to_float(a)), as_bits(f16_to_float(b)),
                                                as_bits(f16_to_float(c))));

  uint16_t result = half_fma_narrow(value, saturate);
  uint32_t omod = (mode >> 6) & 3;
  if (omod) {
    uint16_t magnitude = result & 0x7fff;
    if (magnitude < 0x0400 || (magnitude == 0x0400 && std::abs(value) < 0x1p-14 - 0x1p-26))
      result = 0;
    else if (omod == 3 && magnitude < 0x0800)
      result &= 0x8000;
    else {
      const float scales[] = {1, 2, 4, 0.5f};
      result = goc::float_to_f16(goc::f16_to_float(result) * scales[omod], saturate);
    }
  }
  return mode & GOC_ALU_CLAMP ? half_fma_clamp(result) : result;
}

} // namespace goc

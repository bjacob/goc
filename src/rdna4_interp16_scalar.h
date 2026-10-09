// SPDX-License-Identifier: MIT

#pragma once

#include "internal.h"
#include "rdna4_fp64.h"
#include "rdna4_mixed_fma_scalar.h"

#include <cmath>
#include <stdint.h>

namespace goc {

// Preserve the direction of discarded bits for a later narrowing conversion.
// The product of these FP32 inputs is exact in FP64. TwoSum and round-to-odd
// follow rocjitsu's mixed FMA implementation.
inline double interp16_round_odd(uint32_t a, uint32_t b, uint32_t c) {
  double product = mixed_fma_widen(a) * mixed_fma_widen(b), addend = mixed_fma_widen(c);
  double value = product + addend, virtual_c = value - product;
  double error = (product - (value - virtual_c)) + (addend - virtual_c);
  uint64_t bits = double_bits(value), eb = double_bits(error);
  if (error != 0 && !(bits & 1))
    bits += ((bits ^ eb) >> 63) ? UINT64_MAX : 1;
  return as_double(bits);
}

inline float interp16_rtz_float(uint32_t a, uint32_t b, uint32_t c) {
  if ((a & 0x7fffffff) >= 0x7f800000 || (b & 0x7fffffff) >= 0x7f800000 ||
      (c & 0x7fffffff) >= 0x7f800000)
    return std::fma(as_float(a), as_float(b), as_float(c));
  double value = interp16_round_odd(a, b, c);
  float rounded = float(value);
  if (std::abs(double(rounded)) > std::abs(value))
    rounded = as_float(as_bits(rounded) - 1);
  return rounded;
}

inline uint16_t interp16_rtz_half(uint32_t a, uint32_t b, uint32_t c, bool clamp) {
  uint16_t result;
  if ((a & 0x7fffffff) >= 0x7f800000 || (b & 0x7fffffff) >= 0x7f800000 ||
      (c & 0x7fffffff) >= 0x7f800000)
    result = mixed_fma_special(a, b, c);
  else {
    uint64_t bits = double_bits(interp16_round_odd(a, b, c));
    unsigned exponent = unsigned((bits >> 52) & 2047);
    uint16_t sign = uint16_t((bits >> 48) & 0x8000);
    uint64_t significand = (bits & UINT64_C(0xfffffffffffff)) | (UINT64_C(1) << 52);
    if (exponent < 999)
      result = sign;
    else if (exponent > 1038)
      result = sign | 0x7bff;
    else if (exponent < 1009)
      result = sign | uint16_t(significand >> (1051 - exponent));
    else
      result = sign | uint16_t(((exponent - 1009) << 10) + (significand >> 42));
  }
  return clamp ? half_fma_clamp(result) : result;
}

} // namespace goc

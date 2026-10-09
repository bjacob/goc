// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <algorithm>
#include <array>
#include <stdint.h>

namespace goc_test::mixed_fma_reference {

// Exact integer magnitudes on a 2^-298 grid cover every finite FP32 product
// and addend. This oracle performs no host floating-point arithmetic.
using Magnitude = std::array<uint64_t, 10>;

inline Magnitude shifted(uint64_t value, unsigned shift) {
  Magnitude out = {};
  out[shift / 64] = value << (shift % 64);
  if (shift % 64)
    out[shift / 64 + 1] = value >> (64 - shift % 64);
  return out;
}

inline int compare(const Magnitude &a, const Magnitude &b) {
  for (int i = 9; i >= 0; --i)
    if (a[i] != b[i])
      return a[i] > b[i] ? 1 : -1;
  return 0;
}

inline Magnitude add(const Magnitude &a, const Magnitude &b, bool subtract) {
  Magnitude out;
  uint64_t carry = 0;
  for (int i = 0; i < 10; ++i) {
    uint64_t first = subtract ? a[i] - b[i] : a[i] + b[i];
    out[i] = subtract ? first - carry : first + carry;
    carry = subtract ? (a[i] < b[i] || first < carry) : (first < a[i] || out[i] < first);
  }
  return out;
}

inline bool bit(const Magnitude &a, int position) {
  return position >= 0 && ((a[unsigned(position) / 64] >> (unsigned(position) % 64)) & 1);
}

inline uint32_t pack(const Magnitude &a, bool negative, bool half, bool saturate,
                     bool rtz = false) {
  uint32_t sign = negative ? (half ? 0x8000 : 0x80000000) : 0;
  int top = 639;
  while (top >= 0 && !bit(a, top))
    --top;
  if (top < 0)
    return sign;
  int precision = half ? 11 : 24, min_exp = half ? -14 : -126, max_exp = half ? 15 : 127;
  int exponent = top - 298;
  int shift = std::max(top - precision + 1, min_exp - precision + 1 + 298);
  uint32_t significand = 0;
  for (int i = top; i >= shift; --i)
    significand = (significand << 1) | unsigned(bit(a, i));
  bool sticky = false;
  for (int i = 0; i < (shift - 1) / 64; ++i)
    sticky |= a[i] != 0;
  int tail = (shift - 1) % 64;
  if (tail)
    sticky |= (a[(shift - 1) / 64] & ((UINT64_C(1) << tail) - 1)) != 0;
  if (!rtz)
    significand += bit(a, shift - 1) && (sticky || (significand & 1));
  if (significand == (uint32_t(1) << precision)) {
    significand >>= 1;
    ++exponent;
  }
  if (exponent > max_exp)
    return sign | (half ? ((saturate || rtz) ? 0x7bff : 0x7c00) : (rtz ? 0x7f7fffff : 0x7f800000));
  if (exponent < min_exp)
    return sign | significand;
  return sign | (uint32_t(exponent + (half ? 15 : 127)) << (precision - 1)) |
         (significand & ((uint32_t(1) << (precision - 1)) - 1));
}

struct Number {
  uint64_t significand;
  int exponent;
  bool negative, infinity;
  uint32_t nan;
};

inline Number decode(uint32_t raw, uint32_t mode, int operand) {
  bool half = mode & (GOC_MIX_F16_A << operand);
  if (half)
    raw = uint16_t(raw >> (mode & (GOC_ALU_HIGH_A << operand) ? 16 : 0));
  unsigned fraction_bits = half ? 10 : 23, exponent_bits = half ? 5 : 8;
  uint32_t fraction = raw & ((uint32_t(1) << fraction_bits) - 1);
  unsigned exponent = (raw >> fraction_bits) & ((1u << exponent_bits) - 1);
  bool negative = (raw >> (half ? 15 : 31)) & 1;
  if (mode & (GOC_ALU_ABS_A << operand))
    negative = false;
  if (mode & (GOC_ALU_NEG_A << operand))
    negative = !negative;
  bool special = exponent == ((1u << exponent_bits) - 1);
  return {fraction + (exponent && !special ? UINT64_C(1) << fraction_bits : 0),
          int(exponent ? exponent : 1) - (half ? 25 : 150), negative, special && !fraction,
          special && fraction
              ? (uint32_t(negative) << 31) | 0x7fc00000 | (half ? fraction << 13 : fraction)
              : 0};
}

inline uint32_t evaluate(int output, uint32_t a, uint32_t b, uint32_t c, uint32_t before,
                         uint32_t mode, bool saturate, bool rtz = false) {
  auto x = decode(a, mode, 0), y = decode(b, mode, 1), z = decode(c, mode, 2);
  uint32_t special = 0;
  if ((!x.significand && !x.infinity && !x.nan && y.infinity) ||
      (!y.significand && !y.infinity && !y.nan && x.infinity))
    special = 0xffc00000;
  else if (x.nan || y.nan || z.nan)
    special = x.nan ? x.nan : y.nan ? y.nan : z.nan;
  else if (x.infinity || y.infinity) {
    special = (uint32_t(x.negative != y.negative) << 31) | 0x7f800000;
    if (z.infinity && ((special >> 31) != unsigned(z.negative)))
      special = 0xffc00000;
  } else if (z.infinity)
    special = (uint32_t(z.negative) << 31) | 0x7f800000;
  uint32_t result;
  if (special) {
    result = output ? ((special >> 16) & 0x8000) | ((special >> 13) & 0x3ff) | 0x7c00 : special;
  } else {
    auto product = shifted(x.significand * y.significand, unsigned(x.exponent + y.exponent + 298));
    auto addend = shifted(z.significand, unsigned(z.exponent + 298));
    bool negative = x.negative != y.negative;
    Magnitude sum;
    if (negative == z.negative)
      sum = add(product, addend, false);
    else if (compare(product, addend) > 0)
      sum = add(product, addend, true);
    else {
      negative = compare(product, addend) < 0 ? z.negative : false;
      sum = add(addend, product, true);
    }
    result = pack(sum, negative, output != 0, saturate, rtz);
  }
  if (mode & GOC_ALU_CLAMP) {
    uint32_t sign = output ? 0x8000 : 0x80000000;
    uint32_t infinity = output ? 0x7c00 : 0x7f800000, one = output ? 0x3c00 : 0x3f800000;
    result = (result & sign) || (result & ~sign) > infinity ? 0 : std::min(result, one);
  }
  if (!output)
    return result;
  unsigned shift = output == 2 ? 16 : 0;
  return (before & ~(uint32_t(65535) << shift)) | (result << shift);
}

inline uint32_t mode(unsigned index) {
  return (index & 63) | ((index & 64) ? GOC_ALU_CLAMP : 0) | (((index >> 7) & 7) << 9) |
         (((index >> 10) & 7) << 13);
}

} // namespace goc_test::mixed_fma_reference

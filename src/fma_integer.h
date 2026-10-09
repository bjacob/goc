// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc {

// Shift right, retaining a sticky low bit for all discarded nonzero bits.
inline uint64_t fma_shift_jam(uint64_t value, unsigned shift) {
  if (!shift)
    return value;
  if (shift >= 64)
    return value != 0;
  return (value >> shift) | ((value << (64 - shift)) != 0);
}

// Return a finite FP32 fused product-plus-addend as FP64 bits, rounded to odd.
// Integer arithmetic makes the result independent of host FP controls. The
// extra precision preserves rounding decisions when narrowing to FP32 or FP16.
inline uint64_t fma_finite_round_odd(uint32_t a, uint32_t b, uint32_t c) {
  auto decode = [](uint32_t bits, int &exponent) {
    exponent = int((bits >> 23) & 255);
    uint64_t significand = bits & 0x7fffff;
    if (exponent)
      significand |= 0x800000;
    else
      exponent = 1;
    exponent -= 150;
    return significand;
  };
  int ea, eb, ec;
  uint64_t ma = decode(a, ea), mb = decode(b, eb), mc = decode(c, ec);
  uint64_t product = ma * mb;
  int ep = ea + eb;
  bool negative = ((a ^ b) >> 31) != 0, c_negative = (c >> 31) != 0;
  // Normalize both terms to bit 61, leaving room for a carry and retaining
  // every product bit. Cancellation is exact when the terms are close.
  auto normalize = [](uint64_t &value, int &exponent) {
    if (value)
      while (!(value & (1ULL << 61))) {
        value <<= 1;
        --exponent;
      }
  };
  normalize(product, ep);
  normalize(mc, ec);
  int exponent = !product ? ec : !mc ? ep : ep > ec ? ep : ec;
  if (product)
    product = fma_shift_jam(product, unsigned(exponent - ep));
  if (mc)
    mc = fma_shift_jam(mc, unsigned(exponent - ec));
  uint64_t sum;
  if (negative == c_negative)
    sum = product + mc;
  else if (product > mc)
    sum = product - mc;
  else {
    sum = mc - product;
    negative = mc > product ? c_negative : false;
  }
  if (!sum)
    return uint64_t(negative) << 63;
  if (sum & (1ULL << 62)) {
    sum = fma_shift_jam(sum, 1);
    ++exponent;
  }
  normalize(sum, exponent);
  uint64_t significand = fma_shift_jam(sum, 9);
  return (uint64_t(negative) << 63) | (uint64_t(exponent + 61 + 1023) << 52) |
         (significand & 0xfffffffffffffULL);
}

} // namespace goc

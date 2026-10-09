// SPDX-License-Identifier: MIT

#pragma once

#include "uint128.h"

#include <stdint.h>

namespace goc_test {

struct Mad64Result {
  uint64_t value;
  bool carry;
};

// Compute the complete extended sum, then extract bit 64 and apply saturation.
inline Mad64Result mad64_reference(bool is_signed, uint32_t a, uint32_t b, uint64_t c, bool clamp) {
  using goc::Uint128;
  Uint128 product, accumulator(c);
  if (is_signed) {
    uint64_t am = a >> 31 ? uint32_t(0u - a) : a, bm = b >> 31 ? uint32_t(0u - b) : b;
    product = am * bm;
    if ((a ^ b) >> 31)
      product = Uint128(0) - product;
    accumulator = Uint128(c, c >> 63 ? UINT64_MAX : 0);
  } else
    product = uint64_t(a) * b;
  Uint128 sum = product + accumulator;
  uint64_t value = uint64_t(sum);
  bool carry = (uint64_t(sum >> 64) & 1) != 0;
  bool overflow = is_signed ? (carry != bool(value >> 63)) : carry;
  if (clamp && overflow)
    value = is_signed ? (carry ? UINT64_C(1) << 63 : UINT64_MAX >> 1) : UINT64_MAX;
  return {value, carry};
}

} // namespace goc_test

// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <algorithm>
#include <stdint.h>

namespace goc_test {

inline const char *const dpp_integer_mul_names[] = {"v_mul_i32_i24", "v_mul_hi_i32_i24",
                                                    "v_mul_u32_u24", "v_mul_hi_u32_u24"};

using IntegerMulFn = decltype(&goc_v_mul_lo_u32);
inline const IntegerMulFn integer_mul_functions[] = {
    goc_v_mul_lo_u32,     goc_v_mul_hi_u32,  goc_v_mul_hi_i32,    goc_v_mul_i32_i24,
    goc_v_mul_hi_i32_i24, goc_v_mul_u32_u24, goc_v_mul_hi_u32_u24};

inline bool integer_mul_can_clamp(int op) { return op == 3 || op == 5; }

inline uint32_t integer_mul_reference(int op, uint32_t a, uint32_t b, bool clamp) {
  const bool signed_op = op == 2 || op == 3 || op == 4;
  const bool high = op == 1 || op == 2 || op == 4 || op == 6;
  const int width = op < 3 ? 32 : 24;
  uint64_t modulus = 1ULL << width;
  a %= modulus;
  b %= modulus;
  bool negative_a = signed_op && a >= modulus / 2;
  bool negative_b = signed_op && b >= modulus / 2;
  uint64_t magnitude_a = negative_a ? modulus - a : a;
  uint64_t magnitude_b = negative_b ? modulus - b : b;
  // Multiply magnitudes; restore the sign after applying the range limit.
  uint64_t product = magnitude_a * magnitude_b;
  bool negative = negative_a != negative_b;
  if (clamp)
    product = std::min<uint64_t>(product, signed_op ? (negative ? 0x80000000ULL : 0x7fffffffULL)
                                                    : uint64_t(UINT32_MAX));
  if (negative)
    product = 0ULL - product;
  return uint32_t(high ? product >> 32 : product);
}

} // namespace goc_test

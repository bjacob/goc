// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file rdna4_dot.h
/// @brief Hardware-characterized GFX12 DOT2 and WMMA arithmetic with FP32 or packed outputs.

#include "rdna4_packed16.h"

#include <algorithm>
#include <array>
#include <stdint.h>

// Adapted from rocjitsu shared/gfx12_dot.h. This is the existing gfx1201
// empirical model, adapted to C++17. No new hardware model.
namespace goc {

inline int bit_width(uint64_t x) {
  int n = 0;
  while (x) {
    ++n;
    x >>= 1;
  }
  return n;
}

namespace gfx12_dot_detail {

inline constexpr uint32_t kFactorNan = 0xffc00a3d;
inline constexpr uint32_t kInvalidProductNan = 0xffc00000;
inline constexpr uint32_t kQuietNanBit = 0x00400000;

inline constexpr int kEmptyExponent = -1024;

// A nonnegative integer significand times 2^exponent. Keeping products in this
// representation also preserves BF16 products outside the FP32 exponent range.
struct Term {
  uint64_t significand;
  int exponent;
  bool negative;

  int leading_exponent() const {
    return significand ? exponent + int(bit_width(significand)) - 1 : kEmptyExponent;
  }
};

template <bool Bf16> struct Factor {
  static constexpr int fraction_bits = Bf16 ? 7 : 10;
  static constexpr int bias = Bf16 ? 127 : 15;
  static constexpr uint16_t fraction_mask = (1u << fraction_bits) - 1;
  static constexpr uint16_t infinity = Bf16 ? 0x7f80 : 0x7c00;

  uint16_t bits;

  bool nan() const { return (bits & 0x7fff) > infinity; }

  bool inf() const { return (bits & 0x7fff) == infinity; }

  bool negative() const { return bits >> 15; }

  int exponent_field() const { return (bits & 0x7fff) >> fraction_bits; }

  uint64_t significand() const {
    if (exponent_field())
      return (1u << fraction_bits) | (bits & fraction_mask);
    return bits & fraction_mask;
  }

  int alignment_exponent() const { return int(std::max(exponent_field(), 1)) - bias; }

  Term product(Factor other) const {
    return {significand() * other.significand(),
            alignment_exponent() + other.alignment_exponent() - 2 * fraction_bits,
            negative() != other.negative()};
  }
};

inline uint64_t shift_round_even(uint64_t value, int shift) {
  if (shift <= 0)
    return value << -shift;
  if (shift >= 64)
    return 0;
  const uint64_t tail = value & ((uint64_t{1} << shift) - 1);
  const uint64_t halfway = uint64_t{1} << (shift - 1);
  return (value >> shift) + (tail > halfway || (tail == halfway && ((value >> shift) & 1)));
}

inline uint32_t pack(int64_t units, int grid) {
  if (!units)
    return 0;
  const uint32_t sign = uint32_t(units < 0) << 31;
  const uint64_t magnitude = units < 0 ? uint64_t(-units) : uint64_t(units);
  const int top = int(bit_width(magnitude)) - 1;
  int exponent = top + grid;

  // Round directly on the subnormal grid to avoid double rounding.
  if (exponent < -126) {
    const uint64_t fraction = shift_round_even(magnitude, -149 - grid);
    return fraction ? sign | uint32_t(fraction) : 0;
  }
  uint64_t significand = shift_round_even(magnitude, top - 23);
  if (significand == 0x1000000) {
    significand >>= 1;
    ++exponent;
  }
  if (exponent > 127)
    return sign | 0x7f800000u;
  return sign | (uint32_t(exponent + 127) << 23) | (uint32_t(significand) & 0x7fffff);
}

// Alignment discards low bits by arithmetic floor. An accumulator too small
// to retain any magnitude bits contributes zero, including when negative.
inline int64_t align(Term term, int grid, bool accumulator = false) {
  const int shift = term.exponent - grid;
  const uint64_t magnitude = !term.significand || shift <= -64 ? 0
                             : shift < 0                       ? term.significand >> -shift
                                                               : term.significand << shift;
  const bool tail = term.significand && shift < 0 &&
                    (shift <= -64 || (term.significand & ((uint64_t{1} << -shift) - 1)));
  return term.negative ? -int64_t(magnitude) - (tail && (!accumulator || magnitude))
                       : int64_t(magnitude);
}

} // namespace gfx12_dot_detail

// GFX12 first aligns each pair of products, then aligns pair sums with C.
// Four-product steps are rounded to the output format. Integer arithmetic preserves the
// empirically observed NaN precedence and subnormal behavior.
template <bool Bf16, int N, bool Packed = false>
inline uint32_t gfx12_dot_bits(const std::array<uint16_t, N> &a, const std::array<uint16_t, N> &b,
                               uint32_t acc, bool fp16_ovfl = false) {
  static_assert(N == 2 || N == 4);
  using namespace gfx12_dot_detail;
  const auto special = [](uint32_t bits) -> uint32_t {
    if constexpr (Packed)
      return packed16::special<Bf16>(bits);
    return bits;
  };
  std::array<Term, N> products;
  std::array<int, N> product_grids;
  int product_grid = kEmptyExponent;
  bool positive_inf = false;
  bool negative_inf = false;
  bool invalid = false;
  for (int i = 0; i < N; ++i) {
    const Factor<Bf16> left{a[i]}, right{b[i]};
    if (left.nan() || right.nan())
      return special(kFactorNan);
    invalid |= (left.inf() && !right.significand()) || (right.inf() && !left.significand());
    if (left.inf() || right.inf()) {
      if (left.negative() != right.negative())
        negative_inf = true;
      else
        positive_inf = true;
    }
    products[i] = left.product(right);
    product_grids[i] = products[i].significand
                           ? left.alignment_exponent() + right.alignment_exponent() - 24
                           : kEmptyExponent;
    product_grid = std::max(product_grid, product_grids[i]);
  }

  // Factor NaNs, then invalid products, precede an accumulator NaN.
  if (invalid || (positive_inf && negative_inf))
    return special(kInvalidProductNan);
  if ((acc & 0x7fffffff) > 0x7f800000)
    return special(acc | kQuietNanBit);
  positive_inf |= acc == 0x7f800000;
  negative_inf |= acc == 0xff800000;
  if (positive_inf || negative_inf)
    return special(positive_inf && negative_inf ? kInvalidProductNan
                   : negative_inf               ? 0xff800000
                                                : 0x7f800000);

  const int acc_exp = (acc >> 23) & 255;
  const Term c{acc_exp ? (acc & 0x7fffff) | 0x800000u : acc & 0x7fffff,
               int(std::max(acc_exp, 1)) - 127 - 23, bool(acc >> 31)};
  const int grid =
      std::max(product_grid, std::max(Packed && !Bf16 ? -14 : -126, c.leading_exponent()) - 26);
  int64_t total = align(c, grid, !(Packed && !Bf16 && grid <= -14));
  for (int i = 0; i < N; i += 2) {
    const int pair_grid = std::max(product_grids[i], product_grids[i + 1]);
    const int64_t pair = align(products[i], pair_grid) + align(products[i + 1], pair_grid);
    const Term sum{uint64_t(pair < 0 ? -pair : pair), pair_grid, pair < 0};
    total += align(sum, grid);
  }

  // Pair alignment retains at most 27 magnitude bits; the final sum and C
  // fit comfortably in int64_t. Zero, including underflow to zero, is positive.
  if constexpr (Packed) {
    if constexpr (Bf16)
      return pack(total, grid) >> 16;
    return packed16::pack_f16(total, grid, false, fp16_ovfl);
  }
  return pack(total, grid);
}

} // namespace goc

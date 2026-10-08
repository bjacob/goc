// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <stdint.h>

namespace goc {

// Lane mapping and final-only saturation follow rocjitsu shared/mma_exec.h.
// Widen factors to signed 16 bits, including unsigned bytes (0..255), before
// pairwise dot products. No intermediate byte-pair saturation is permitted.
template <int Bits, int K, class Ops>
void integer_wmma(uint32_t mask, uint32_t modifiers, uint32_t *const *d, const uint32_t *const *a,
                  const uint32_t *const *b, const uint32_t *const *c) {
  using V = typename Ops::V;
  const int sign_a = (modifiers & GOC_WMMA_SIGNED_A) ? 1 << (Bits - 1) : 0;
  const int sign_b = (modifiers & GOC_WMMA_SIGNED_B) ? 1 << (Bits - 1) : 0;
  const auto location = [](int k, int &reg, int &lane, int &shift) {
    int local = k % 8 + 8 * (k / 16);
    reg = local * Bits / 32;
    lane = 16 * ((k / 8) % 2);
    shift = local * Bits % 32;
  };
  uint32_t packed[2][K / 2][16];
  for (int operand = 0; operand < 2; ++operand) {
    const uint32_t *const *v = operand ? b : a;
    int sign = operand ? sign_b : sign_a;
    for (int k = 0; k < K; k += 2) {
      int reg, lane, shift;
      location(k, reg, lane, shift);
      for (int index = 0; index < 16; index += Ops::width) {
        V words = Ops::load(v[reg] + lane + index);
        const auto extend = [&](int offset) {
          V value = Ops::bit_and(Ops::shift_right(words, offset), Ops::splat((1 << Bits) - 1));
          return Ops::sub(Ops::bit_xor(value, Ops::splat(sign)), Ops::splat(sign));
        };
        V pair = Ops::bit_or(Ops::bit_and(extend(shift), Ops::splat(65535)),
                             Ops::shift_left16(extend(shift + Bits)));
        Ops::store(packed[operand][k / 2] + index, pair);
      }
    }
  }

  uint32_t result[8][32];
  for (int row = 0; row < 16; ++row) {
    V sums[16 / Ops::width];
    for (auto &sum : sums)
      sum = Ops::splat(0);
    for (int k = 0; k < K; k += 2) {
      V left = Ops::splat_bits(packed[0][k / 2][row]);
      for (int chunk = 0; chunk < 16 / Ops::width; ++chunk)
        sums[chunk] = Ops::dot(sums[chunk], left, Ops::load(packed[1][k / 2] + chunk * Ops::width));
    }
    for (int chunk = 0; chunk < 16 / Ops::width; ++chunk) {
      int lane = 16 * (row / 8) + chunk * Ops::width;
      V acc = Ops::load(c[row % 8] + lane);
      V sum = Ops::add(acc, sums[chunk]);
      // The dot alone fits int32 (at most 16*255*255). Only adding C can
      // overflow. Detect signed overflow and clamp once, after the entire dot.
      if (modifiers & GOC_WMMA_CLAMP) {
        V overflow = Ops::bit_and(Ops::bit_xor(acc, sum), Ops::bit_xor(sums[chunk], sum));
        V limit = Ops::bit_xor(Ops::sign(acc), Ops::splat(INT32_MAX));
        sum = Ops::select_negative(overflow, limit, sum);
      }
      Ops::store(result[row % 8] + lane, sum);
    }
  }
  // All source reads precede writes, including when D shares A/B/C VGPRs.
  for (int reg = 0; reg < 8; ++reg)
    for (int lane = 0; lane < 32; lane += Ops::width)
      Ops::masked_store(d[reg] + lane, mask >> lane, Ops::load(result[reg] + lane));
}

} // namespace goc

// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Layouts adapted from rocjitsu shared/mma_exec.h. Intermediate CLAMP stages
// follow gfx1201 captures, including the interleaved K=64 product groups.

#pragma once

#include "goc/goc.h"

#include <stdint.h>

namespace goc {

struct SwmmacIntegerInputs {
  int32_t a[16][32];
  int32_t b[64][16];
  int32_t acc[16][16];
  uint8_t selected[16][32];
};

// Snapshot all input and accumulator registers into logical matrix order.
template <unsigned Bits, unsigned K>
inline void swmmac_integer_prepare(SwmmacIntegerInputs &input, uint32_t mode, uint32_t *const *d,
                                   const uint32_t *const *a, const uint32_t *const *b,
                                   const uint32_t *const *index) {
  auto extend = [](uint32_t x, bool sign) {
    return int32_t(x) - (sign && (x & (1u << (Bits - 1))) ? int32_t(1u << Bits) : 0);
  };
  unsigned key = mode & GOC_SWMMAC_INDEX_KEY_1 ? 16 : 0;
  for (unsigned row = 0; row < 16; ++row)
    for (unsigned ck = 0; ck < K / 2; ++ck) {
      unsigned lane = row + 16 * ((ck / 8) & 1), local = 8 * (ck / 16) + ck % 8;
      uint32_t raw = (a[local * Bits / 32][lane] >> (local * Bits % 32)) & ((1u << Bits) - 1);
      input.a[row][ck] = extend(raw, mode & GOC_WMMA_SIGNED_A);
      input.selected[row][ck] = uint8_t(4 * (ck / 2) + ((index[0][lane] >> (key + 2 * local)) & 3));
    }
  for (unsigned k = 0; k < K; ++k)
    for (unsigned col = 0; col < 16; ++col) {
      unsigned lane = col + 16 * ((k / 16) & 1), local = 16 * (k / 32) + k % 16;
      uint32_t raw = (b[local * Bits / 32][lane] >> (local * Bits % 32)) & ((1u << Bits) - 1);
      input.b[k][col] = extend(raw, mode & GOC_WMMA_SIGNED_B);
    }
  for (unsigned row = 0; row < 16; ++row)
    for (unsigned col = 0; col < 16; ++col) {
      uint32_t raw = d[row % 8][col + 16 * (row / 8)];
      input.acc[row][col] = int32_t(int64_t(raw) - ((raw >> 31) ? (INT64_C(1) << 32) : 0));
    }
}

// Return the compressed product position within one hardware accumulation stage.
inline unsigned swmmac_integer_ck(unsigned stage, unsigned local) {
  return 16 * (local / 8) + 8 * stage + local % 8;
}

void swmmac_integer_x86_64_v3(unsigned k, uint32_t mask, bool clamp, uint32_t *const *d,
                              const SwmmacIntegerInputs &input);
void swmmac_integer_x86_64_v4(unsigned k, uint32_t mask, bool clamp, uint32_t *const *d,
                              const SwmmacIntegerInputs &input);

} // namespace goc

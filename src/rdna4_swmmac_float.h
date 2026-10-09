// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Wave32 sparse layouts adapted from rocjitsu shared/mma_exec.h. Negation
// follows gfx1201 captures: B modifiers act on the selected pair positions.

#pragma once

#include "float_formats.h"
#include "goc/goc.h"
#include "internal.h"

#include <stdint.h>

namespace goc {

struct SwmmacFloatInputs {
  float a[16][16];
  float b[32][16];
  float acc[16][16];
  uint8_t selected[16][16];
};

// Accumulate all decoded products in sparse K order, applying B negation by pair position.
// Updates only input.acc; input arrays must already include any A modifiers.
void swmmac_float_accumulate(uint32_t mode, SwmmacFloatInputs &input);

// Decode all input registers and the initial accumulator before destination
// writes. Each metadata pair must contain two increasing positions in [0,3].
template <bool Bf16, bool Packed>
inline void swmmac16_prepare(SwmmacFloatInputs &input, uint32_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *index) {
  auto decode = [](uint16_t x) { return Bf16 ? bf16_to_float(x) : f16_to_float(x); };
  unsigned key = mode & GOC_SWMMAC_INDEX_KEY_1 ? 16 : 0;
  for (unsigned row = 0; row < 16; ++row)
    for (unsigned ck = 0; ck < 16; ++ck) {
      unsigned group = ck / 2, slot = ck % 2, lane = row + 16 * ((group / 2) & 1),
               reg = 2 * (group / 4) + (group & 1);
      uint16_t value = uint16_t(a[reg][lane] >> (16 * slot));
      if (mode & (slot ? GOC_WMMA_NEG_HI_A : GOC_WMMA_NEG_LO_A))
        value ^= 0x8000;
      input.a[row][ck] = decode(value);
      unsigned local = 2 * (group & 1) + 4 * (group / 4) + slot;
      input.selected[row][ck] = uint8_t(4 * group + ((index[0][lane] >> (key + 2 * local)) & 3));
    }
  for (unsigned k = 0; k < 32; ++k)
    for (unsigned col = 0; col < 16; ++col) {
      unsigned lane = col + 16 * ((k / 8) & 1), slot = 8 * (k / 16) + 2 * ((k / 2) & 3) + (k & 1);
      input.b[k][col] = decode(uint16_t(b[slot / 2][lane] >> (16 * (slot % 2))));
    }
  for (unsigned row = 0; row < 16; ++row)
    for (unsigned col = 0; col < 16; ++col) {
      unsigned reg = row % 8, lane = col + 16 * (row / 8);
      if constexpr (Packed)
        input.acc[row][col] = decode(uint16_t(d[reg / 2][lane] >> (16 * (reg % 2))));
      else
        input.acc[row][col] = as_float(d[reg][lane]);
    }
}

// Pack a complete result matrix into physical destination words.
template <bool Bf16, bool Packed>
inline void swmmac_float_pack(uint32_t (&result)[Packed ? 4 : 8][32], const float (&acc)[16][16],
                              bool saturate) {
  for (unsigned row = 0; row < 16; ++row)
    for (unsigned col = 0; col < 16; ++col) {
      unsigned reg = row % 8, lane = col + 16 * (row / 8);
      if constexpr (Packed) {
        uint16_t value =
            Bf16 ? float_to_bf16(acc[row][col]) : float_to_f16(acc[row][col], saturate);
        if (reg % 2)
          result[reg / 2][lane] |= uint32_t(value) << 16;
        else
          result[reg / 2][lane] = value;
      } else
        result[reg][lane] = as_bits(acc[row][col]);
    }
}

template <bool Bf16, bool Packed>
void swmmac_float_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                            SwmmacFloatInputs &input, bool saturate);

template <bool Bf16, bool Packed>
void swmmac_float_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                            SwmmacFloatInputs &input, bool saturate);

} // namespace goc

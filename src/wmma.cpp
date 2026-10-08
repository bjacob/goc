// SPDX-License-Identifier: MIT
#include "float_formats.h"
#include "rdna4_dot.h"
#include <cmath>
namespace {
// Physical packing follows rocjitsu shared/mma_exec.h: each lane supplies
// eight consecutive K elements. Output lanes select columns and 8-row groups.
template <bool Bf16>
int wmma(uint64_t flags, uint64_t mask, uint32_t instruction_flags, uint32_t *const *d,
         uint32_t *const *a, uint32_t *const *b, uint32_t *const *c) {
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result[8][32];
  const auto read_bits = [](uint32_t *const *v, int index, int k) {
    return uint16_t(v[(k % 8) / 2][index + 16 * (k / 8)] >> (16 * (k % 2)));
  };
  for (int row = 0; row < 16; ++row)
    for (int col = 0; col < 16; ++col) {
      int lane = col + 16 * (row / 8), reg = row % 8;
      if (!((mask >> lane) & 1))
        continue;
      if ((flags & GOC_SEMANTICS_MASK) == GOC_SEMANTICS_EXACT) {
        uint32_t acc = c[reg][lane];
        for (int k = 0; k < 16; k += 4) {
          std::array<uint16_t, 4> left, right;
          for (int j = 0; j < 4; ++j) {
            // GFX12 wave32 processes physical K chunks in order 0,8,4,12.
            int logical = k + j;
            int physical = (logical & 3) | ((logical & 4) << 1) | ((logical & 8) >> 1);
            left[j] = read_bits(a, row, physical);
            right[j] = read_bits(b, col, physical);
          }
          acc = goc::gfx12_dot_bits<Bf16, 4>(left, right, acc);
        }
        result[reg][lane] = acc;
      } else {
        float acc = goc::as_float(c[reg][lane]);
        for (int k = 0; k < 16; ++k) {
          auto left = read_bits(a, row, k), right = read_bits(b, col, k);
          float x = Bf16 ? goc::bf16_to_float(left) : goc::f16_to_float(left);
          float y = Bf16 ? goc::bf16_to_float(right) : goc::f16_to_float(right);
          acc = std::fma(x, y, acc);
        }
        result[reg][lane] = goc::as_bits(acc);
      }
    }
  // Delayed stores are necessary even with exact whole-VGPR aliasing: a
  // destination may overwrite sources consumed by a different output lane.
  for (int reg = 0; reg < 8; ++reg)
    for (int lane = 0; lane < 32; ++lane)
      if ((mask >> lane) & 1)
        d[reg][lane] = result[reg][lane];
  return GOC_SUCCESS;
}
} // namespace
int goc_rdna4_v_wmma_f32_16x16x16_f16(uint64_t flags, uint64_t mask, uint32_t instruction_flags,
                                      uint32_t *const *d, uint32_t *const *a, uint32_t *const *b,
                                      uint32_t *const *c) {
  return wmma<false>(flags, mask, instruction_flags, d, a, b, c);
}
int goc_rdna4_v_wmma_f32_16x16x16_bf16(uint64_t flags, uint64_t mask, uint32_t instruction_flags,
                                       uint32_t *const *d, uint32_t *const *a, uint32_t *const *b,
                                       uint32_t *const *c) {
  return wmma<true>(flags, mask, instruction_flags, d, a, b, c);
}

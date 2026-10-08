// SPDX-License-Identifier: MIT
#include "float_formats.h"
#include <cmath>
namespace {
// Physical packing follows rocjitsu shared/mma_exec.h: each lane supplies
// eight consecutive K elements. Output lanes select columns and 8-row groups.
template <bool Bf16>
int wmma(uint64_t flags, uint64_t mask, uint32_t instruction_flags, uint32_t *const *d,
         uint32_t *const *a, uint32_t *const *b, uint32_t *const *c) {
  if (int error = goc::validate(flags, instruction_flags))
    return error;
  uint32_t result[8][32];
  const auto read = [](uint32_t *const *v, int index, int k) {
    uint16_t bits = uint16_t(v[(k % 8) / 2][index + 16 * (k / 8)] >> (16 * (k % 2)));
    return Bf16 ? goc::bf16_to_float(bits) : goc::f16_to_float(bits);
  };
  for (int row = 0; row < 16; ++row)
    for (int col = 0; col < 16; ++col) {
      int lane = col + 16 * (row / 8), reg = row % 8;
      if (!((mask >> lane) & 1))
        continue;
      float acc = goc::as_float(c[reg][lane]);
      for (int k = 0; k < 16; ++k)
        acc = std::fma(read(a, row, k), read(b, col, k), acc);
      result[reg][lane] = goc::as_bits(acc);
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

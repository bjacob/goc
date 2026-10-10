// SPDX-License-Identifier: MIT

#include "float_formats.h"
#include "goc/goc.h"
#include "internal.h"

#include <cmath>
#include <stdint.h>

namespace {

// RDNA3 layouts follow rocjitsu shared/mma_exec.h's gfx11_wmma_input_loc
// and gfx11_wmma_output_loc_32. Each 16-lane group holds the full A/B tile;
// output rows are interleaved across groups, unlike RDNA4's blocked layout.
template <bool Bf16, int Lanes>
int wmma(uint64_t flags, uint64_t mode, uint32_t *const *d, const uint32_t *const *a,
         const uint32_t *const *b, const uint32_t *const *c) {
  if (mode & ~(uint64_t(GOC_WMMA_NEG_C) | GOC_WMMA_ABS_C))
    return GOC_ERROR_INVALID_FLAGS;
  // Do not reuse the GFX12 empirical dot model for a different architecture.
  if (int error = goc::validate(flags, 0))
    return error;
  constexpr int Groups = Lanes / 16;
  constexpr int Registers = 256 / Lanes;
  uint32_t result[Registers][Lanes];
  for (int row = 0; row < 16; ++row)
    for (int col = 0; col < 16; ++col) {
      int group = row % Groups, lane = group * 16 + col, reg = row / Groups;
      uint32_t acc_bits = c[reg][lane];
      if (mode & GOC_WMMA_ABS_C)
        acc_bits &= 0x7fffffff;
      if (mode & GOC_WMMA_NEG_C)
        acc_bits ^= 0x80000000;
      float acc = goc::as_float(acc_bits);
      for (int k = 0; k < 16; ++k) {
        uint16_t av = uint16_t(a[k / 2][group * 16 + row] >> (16 * (k % 2)));
        uint16_t bv = uint16_t(b[k / 2][group * 16 + col] >> (16 * (k % 2)));
        float x = Bf16 ? goc::bf16_to_float(av) : goc::f16_to_float(av);
        float y = Bf16 ? goc::bf16_to_float(bv) : goc::f16_to_float(bv);
        acc = std::fma(x, y, acc);
      }
      result[reg][lane] = goc::as_bits(acc);
    }
  // Stage the whole matrix before writing: operands can share whole VGPRs.
  for (int reg = 0; reg < Registers; ++reg)
    for (int lane = 0; lane < Lanes; ++lane)
      d[reg][lane] = result[reg][lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_v_wmma_f32_16x16x16_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                const uint32_t *const *c) {
  return wmma<false, 32>(flags, instruction_flags, d, a, b, c);
}

int goc_v_wmma_f32_16x16x16_bf16(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                                 const uint32_t *const *a, const uint32_t *const *b,
                                 const uint32_t *const *c) {
  return wmma<true, 32>(flags, instruction_flags, d, a, b, c);
}

int goc_v_wmma_f32_16x16x16_f16_wave64(uint64_t flags, uint64_t instruction_flags,
                                       uint32_t *const *d, const uint32_t *const *a,
                                       const uint32_t *const *b, const uint32_t *const *c) {
  return wmma<false, 64>(flags, instruction_flags, d, a, b, c);
}

int goc_v_wmma_f32_16x16x16_bf16_wave64(uint64_t flags, uint64_t instruction_flags,
                                        uint32_t *const *d, const uint32_t *const *a,
                                        const uint32_t *const *b, const uint32_t *const *c) {
  return wmma<true, 64>(flags, instruction_flags, d, a, b, c);
}

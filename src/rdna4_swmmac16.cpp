// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "internal.h"
#include "rdna4_swmmac_float.h"

#include <cmath>
#include <stdint.h>

namespace {

template <bool Bf16, bool Packed>
int run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *index) {
  const uint32_t known = GOC_WMMA_NEG_LO_A | GOC_WMMA_NEG_LO_B | GOC_WMMA_NEG_HI_A |
                         GOC_WMMA_NEG_HI_B | GOC_SWMMAC_INDEX_KEY_1;
  if (int error = goc::validate(flags, mode & ~known))
    return error;
  uint32_t mask = exec_mask;
  if (!mask)
    return GOC_SUCCESS;
  goc::SwmmacFloatInputs input;
  goc::swmmac16_prepare<Bf16, Packed>(input, mode, d, a, b, index);
  bool saturate = (flags & GOC_FP16_OVFL) != 0;
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::swmmac_float_x86_64_v4<Bf16, Packed>(mask, mode, d, input, saturate);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::swmmac_float_x86_64_v3<Bf16, Packed>(mask, mode, d, input, saturate);
    return GOC_SUCCESS;
  }
#endif
  const uint32_t flip[] = {mode & GOC_WMMA_NEG_LO_B ? 0x80000000u : 0,
                           mode & GOC_WMMA_NEG_HI_B ? 0x80000000u : 0};
  for (unsigned row = 0; row < 16; ++row)
    for (unsigned col = 0; col < 16; ++col) {
      float acc = input.acc[row][col];
      for (unsigned ck = 0; ck < 16; ++ck) {
        float bv =
            goc::as_float(goc::as_bits(input.b[input.selected[row][ck]][col]) ^ flip[ck & 1]);
        acc = std::fma(input.a[row][ck], bv, acc);
      }
      input.acc[row][col] = acc;
    }
  uint32_t result[Packed ? 4 : 8][32];
  goc::swmmac_float_pack<Bf16, Packed>(result, input.acc, saturate);
  for (unsigned reg = 0; reg < (Packed ? 4u : 8u); ++reg)
    for (unsigned lane = 0; lane < 32; ++lane)
      if ((mask >> lane) & 1)
        d[reg][lane] = result[reg][lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_swmmac_f32_16x16x32_f16(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *index) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<false, false>(flags, exec_mask, instruction_flags, d, a, b, index);
}

int goc_rdna4_v_swmmac_f32_16x16x32_bf16(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *index) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<true, false>(flags, exec_mask, instruction_flags, d, a, b, index);
}

int goc_rdna4_v_swmmac_f16_16x16x32_f16(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *index) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<false, true>(flags, exec_mask, instruction_flags, d, a, b, index);
}

int goc_rdna4_v_swmmac_bf16_16x16x32_bf16(uint64_t flags, uint32_t exec_mask,
                                          uint64_t instruction_flags, uint32_t *const *d,
                                          const uint32_t *const *a, const uint32_t *const *b,
                                          const uint32_t *const *index) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<true, true>(flags, exec_mask, instruction_flags, d, a, b, index);
}

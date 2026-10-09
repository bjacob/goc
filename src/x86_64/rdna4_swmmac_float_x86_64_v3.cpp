// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_swmmac_float.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {

template <bool Bf16, bool Packed>
void swmmac_float_x86_64_v3(uint32_t mask, uint32_t mode, uint32_t *const *d,
                            SwmmacFloatInputs &input, bool saturate) {
  const uint32_t flip[] = {mode & GOC_WMMA_NEG_LO_B ? 0x80000000u : 0,
                           mode & GOC_WMMA_NEG_HI_B ? 0x80000000u : 0};
  for (unsigned row = 0; row < 16; ++row)
    for (unsigned col = 0; col < 16; col += 8) {
      auto acc = _mm256_loadu_ps(input.acc[row] + col);
      for (unsigned ck = 0; ck < 16; ++ck) {
        auto bv = _mm256_loadu_ps(input.b[input.selected[row][ck]] + col);
        bv = _mm256_xor_ps(bv, _mm256_castsi256_ps(_mm256_set1_epi32(int(flip[ck & 1]))));
        acc = _mm256_fmadd_ps(_mm256_set1_ps(input.a[row][ck]), bv, acc);
      }
      _mm256_storeu_ps(input.acc[row] + col, acc);
    }
  uint32_t result[Packed ? 4 : 8][32];
  swmmac_float_pack<Bf16, Packed>(result, input.acc, saturate);
  for (unsigned reg = 0; reg < (Packed ? 4u : 8u); ++reg)
    for (unsigned lane = 0; lane < 32; lane += 8) {
      auto value = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(result[reg] + lane));
      auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(mask >> lane)),
                                      _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
      _mm256_maskstore_epi32(reinterpret_cast<int *>(d[reg] + lane), active, value);
    }
}

template void swmmac_float_x86_64_v3<false, false>(uint32_t, uint32_t, uint32_t *const *,
                                                   SwmmacFloatInputs &, bool);
template void swmmac_float_x86_64_v3<false, true>(uint32_t, uint32_t, uint32_t *const *,
                                                  SwmmacFloatInputs &, bool);
template void swmmac_float_x86_64_v3<true, false>(uint32_t, uint32_t, uint32_t *const *,
                                                  SwmmacFloatInputs &, bool);
template void swmmac_float_x86_64_v3<true, true>(uint32_t, uint32_t, uint32_t *const *,
                                                 SwmmacFloatInputs &, bool);

} // namespace goc

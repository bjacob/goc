// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_swmmac16.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {

template <bool Bf16, bool Packed>
void swmmac16_x86_64_v4(uint32_t mask, uint32_t mode, uint32_t *const *d, Swmmac16Inputs &input,
                        bool saturate) {
  const uint32_t flip[] = {mode & GOC_WMMA_NEG_LO_B ? 0x80000000u : 0,
                           mode & GOC_WMMA_NEG_HI_B ? 0x80000000u : 0};
  for (unsigned row = 0; row < 16; ++row)
    for (unsigned col = 0; col < 16; col += 16) {
      auto acc = _mm512_loadu_ps(input.acc[row] + col);
      for (unsigned ck = 0; ck < 16; ++ck) {
        auto bv = _mm512_loadu_ps(input.b[input.selected[row][ck]] + col);
        bv = _mm512_xor_ps(bv, _mm512_castsi512_ps(_mm512_set1_epi32(int(flip[ck & 1]))));
        acc = _mm512_fmadd_ps(_mm512_set1_ps(input.a[row][ck]), bv, acc);
      }
      _mm512_storeu_ps(input.acc[row] + col, acc);
    }
  uint32_t result[Packed ? 4 : 8][32];
  swmmac16_pack<Bf16, Packed>(result, input.acc, saturate);
  for (unsigned reg = 0; reg < (Packed ? 4u : 8u); ++reg)
    for (unsigned lane = 0; lane < 32; lane += 16) {
      auto value = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(result[reg] + lane));
      _mm512_mask_storeu_epi32(d[reg] + lane, __mmask16(mask >> lane), value);
    }
}

template void swmmac16_x86_64_v4<false, false>(uint32_t, uint32_t, uint32_t *const *,
                                               Swmmac16Inputs &, bool);
template void swmmac16_x86_64_v4<false, true>(uint32_t, uint32_t, uint32_t *const *,
                                              Swmmac16Inputs &, bool);
template void swmmac16_x86_64_v4<true, false>(uint32_t, uint32_t, uint32_t *const *,
                                              Swmmac16Inputs &, bool);
template void swmmac16_x86_64_v4<true, true>(uint32_t, uint32_t, uint32_t *const *,
                                             Swmmac16Inputs &, bool);

} // namespace goc

// SPDX-License-Identifier: MIT

#include "../src/internal.h"

#include <immintrin.h>

namespace goc {
void wmma_avx512bf16(uint32_t mask, uint32_t *const *d, uint32_t *const *a, uint32_t *const *b,
                     uint32_t *const *c) {
  uint32_t result[8][32];
  for (int row = 0; row < 16; ++row) {
    int reg = row % 8, group = row / 8;
    __m512 acc = _mm512_castsi512_ps(_mm512_loadu_si512(c[reg] + 16 * group));
    for (int pair = 0; pair < 8; ++pair) {
      uint32_t aw = a[pair % 4][row + 16 * (pair / 4)];
      int32_t signed_word;
      std::memcpy(&signed_word, &aw, sizeof(aw));
      __m512bh va = (__m512bh)_mm512_set1_epi32(signed_word);
      __m512bh vb = (__m512bh)_mm512_loadu_si512(b[pair % 4] + 16 * (pair / 4));
      acc = _mm512_dpbf16_ps(acc, va, vb);
    }
    _mm512_storeu_si512(result[reg] + 16 * group, _mm512_castps_si512(acc));
  }
  for (int reg = 0; reg < 8; ++reg)
    for (int group = 0; group < 2; ++group)
      _mm512_mask_storeu_epi32(d[reg] + 16 * group, static_cast<__mmask16>(mask >> (16 * group)),
                               _mm512_loadu_si512(result[reg] + 16 * group));
}
} // namespace goc

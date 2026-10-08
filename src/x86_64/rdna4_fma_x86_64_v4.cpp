// SPDX-License-Identifier: MIT

#include "rdna4_simd.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {

void fma_x86_64_v4(uint32_t mask, uint32_t *d, const uint32_t *a, const uint32_t *b,
                   const uint32_t *c) {
  for (int i = 0; i < 32; i += 16) {
    auto va = _mm512_castsi512_ps(_mm512_loadu_si512(a + i));
    auto vb = _mm512_castsi512_ps(_mm512_loadu_si512(b + i));
    auto vc = _mm512_castsi512_ps(_mm512_loadu_si512(c + i));
    _mm512_mask_storeu_epi32(d + i, static_cast<__mmask16>(mask >> i),
                             _mm512_castps_si512(_mm512_fmadd_ps(va, vb, vc)));
  }
}

} // namespace goc

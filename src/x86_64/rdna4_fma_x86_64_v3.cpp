// SPDX-License-Identifier: MIT

#include "rdna4_simd.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {

void fma_x86_64_v3(uint32_t mask, uint32_t *d, const uint32_t *a, const uint32_t *b,
                   const uint32_t *c) {
  for (int i = 0; i < 32; i += 8) {
    auto va = _mm256_castsi256_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i)));
    auto vb = _mm256_castsi256_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i)));
    auto vc = _mm256_castsi256_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(c + i)));
    __m256i active = _mm256_setr_epi32(-int((mask >> i) & 1), -int((mask >> (i + 1)) & 1),
                                       -int((mask >> (i + 2)) & 1), -int((mask >> (i + 3)) & 1),
                                       -int((mask >> (i + 4)) & 1), -int((mask >> (i + 5)) & 1),
                                       -int((mask >> (i + 6)) & 1), -int((mask >> (i + 7)) & 1));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + i), active,
                           _mm256_castps_si256(_mm256_fmadd_ps(va, vb, vc)));
  }
}

} // namespace goc

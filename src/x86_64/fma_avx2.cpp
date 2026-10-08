// SPDX-License-Identifier: MIT

#include "internal.h"

#include <immintrin.h>

namespace goc {
void fma_avx2(uint32_t mask, uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  for (int i = 0; i < 32; i += 8) {
    auto va = _mm256_castsi256_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i)));
    auto vb = _mm256_castsi256_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i)));
    auto vc = _mm256_castsi256_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(c + i)));
    uint32_t result[8];
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(result),
                        _mm256_castps_si256(_mm256_fmadd_ps(va, vb, vc)));
    for (int j = 0; j < 8; ++j)
      if ((mask >> (i + j)) & 1)
        d[i + j] = result[j];
  }
}
} // namespace goc

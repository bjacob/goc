// SPDX-License-Identifier: MIT

#include "rdna4_simd.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {

void wmma_f16_x86_64_v3(uint32_t mask, uint32_t *const *d, const uint32_t *const *a,
                        const uint32_t *const *b, const uint32_t *const *c) {
  // Decode B once for all output rows. Each input word holds two consecutive
  // K elements; columns occupy consecutive lanes within each half-wave.
  alignas(32) float right[16][16];
  for (int k = 0; k < 16; ++k)
    for (int col = 0; col < 16; col += 8) {
      __m256i words = _mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(b[(k % 8) / 2] + 16 * (k / 8) + col));
      words =
          k % 2 ? _mm256_srli_epi32(words, 16) : _mm256_and_si256(words, _mm256_set1_epi32(65535));
      __m128i halves =
          _mm_packus_epi32(_mm256_castsi256_si128(words), _mm256_extracti128_si256(words, 1));
      _mm256_store_ps(right[k] + col, _mm256_cvtph_ps(halves));
    }

  uint32_t result[8][32];
  for (int row = 0; row < 16; ++row) {
    int reg = row % 8, lane = 16 * (row / 8);
    __m256 low =
        _mm256_castsi256_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(c[reg] + lane)));
    __m256 high = _mm256_castsi256_ps(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c[reg] + lane + 8)));
    for (int k = 0; k < 16; ++k) {
      uint16_t bits = uint16_t(a[(k % 8) / 2][row + 16 * (k / 8)] >> (16 * (k % 2)));
      __m256 left = _mm256_set1_ps(_cvtsh_ss(bits));
      low = _mm256_fmadd_ps(left, _mm256_load_ps(right[k]), low);
      high = _mm256_fmadd_ps(left, _mm256_load_ps(right[k] + 8), high);
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[reg] + lane), _mm256_castps_si256(low));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[reg] + lane + 8),
                        _mm256_castps_si256(high));
  }

  // Stage all outputs before any stores, since D may overlap A, B or C.
  for (int lane = 0; lane < 32; lane += 8) {
    __m256i active =
        _mm256_setr_epi32(-int((mask >> lane) & 1), -int((mask >> (lane + 1)) & 1),
                          -int((mask >> (lane + 2)) & 1), -int((mask >> (lane + 3)) & 1),
                          -int((mask >> (lane + 4)) & 1), -int((mask >> (lane + 5)) & 1),
                          -int((mask >> (lane + 6)) & 1), -int((mask >> (lane + 7)) & 1));
    for (int reg = 0; reg < 8; ++reg)
      _mm256_maskstore_epi32(
          reinterpret_cast<int *>(d[reg] + lane), active,
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(result[reg] + lane)));
  }
}

} // namespace goc

// SPDX-License-Identifier: MIT

#include "rdna4_dpp.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {

void dpp8_x86_64_v3(uint32_t mask, uint32_t selectors, bool fi, uint32_t *out, const uint32_t *a) {
  auto shifts = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
  auto index = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(int(selectors)), shifts),
                                _mm256_set1_epi32(7));
  for (unsigned lane = 0; lane < 32; lane += 8) {
    auto value = _mm256_permutevar8x32_epi32(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane)), index);
    if (!fi) {
      auto active = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(int(mask >> lane)), index),
                                     _mm256_set1_epi32(1));
      value = _mm256_and_si256(value, _mm256_sub_epi32(_mm256_setzero_si256(), active));
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + lane), value);
  }
}

} // namespace goc

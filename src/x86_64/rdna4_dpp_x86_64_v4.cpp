// SPDX-License-Identifier: MIT

#include "rdna4_dpp.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {

void dpp8_x86_64_v4(uint32_t mask, uint32_t selectors, bool fi, uint32_t *out, const uint32_t *a) {
  auto shifts = _mm512_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21, 0, 3, 6, 9, 12, 15, 18, 21);
  auto index = _mm512_and_si512(_mm512_srlv_epi32(_mm512_set1_epi32(int(selectors)), shifts),
                                _mm512_set1_epi32(7));
  index = _mm512_or_si512(index, _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8));
  for (unsigned lane = 0; lane < 32; lane += 16) {
    auto value = _mm512_permutexvar_epi32(index, _mm512_loadu_si512(a + lane));
    if (!fi) {
      auto active = _mm512_and_si512(_mm512_srlv_epi32(_mm512_set1_epi32(int(mask >> lane)), index),
                                     _mm512_set1_epi32(1));
      value = _mm512_and_si512(value, _mm512_sub_epi32(_mm512_setzero_si512(), active));
    }
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(out + lane), value);
  }
}

} // namespace goc

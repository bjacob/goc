// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {

// Prepare FP32 results for a nonzero OMOD: tiny values become +0, while
// halving normal values below twice minimum normal produces signed zero.
inline __m512 prepare_omod_f32(__m512 value, uint32_t mode) {
  auto magnitude = _mm512_and_si512(_mm512_castps_si512(value), _mm512_set1_epi32(INT32_MAX));
  auto tiny = _mm512_cmplt_epi32_mask(magnitude, _mm512_set1_epi32(0x00800000));
  value = _mm512_mask_mov_ps(value, tiny, _mm512_setzero_ps());
  if ((mode & GOC_ALU_OMOD_HALF) == GOC_ALU_OMOD_HALF) {
    auto underflow = _mm512_cmplt_epi32_mask(magnitude, _mm512_set1_epi32(0x01000000));
    auto sign = _mm512_castsi512_ps(
        _mm512_and_si512(_mm512_castps_si512(value), _mm512_set1_epi32(INT32_MIN)));
    value = _mm512_mask_mov_ps(value, underflow, sign);
  }
  return value;
}

} // namespace goc

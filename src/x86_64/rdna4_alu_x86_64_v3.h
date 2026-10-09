// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {

// Replace FP32 subnormals with zero of the same sign.
inline __m256 flush_denorm_f32(__m256 value) {
  auto bits = _mm256_castps_si256(value);
  auto magnitude = _mm256_and_si256(bits, _mm256_set1_epi32(INT32_MAX));
  auto tiny = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x00800000), magnitude);
  auto sign = _mm256_and_si256(bits, _mm256_set1_epi32(INT32_MIN));
  return _mm256_blendv_ps(value, _mm256_castsi256_ps(sign), _mm256_castsi256_ps(tiny));
}

// Prepare FP32 results for a nonzero OMOD: tiny values become +0, while
// halving normal values below twice minimum normal produces signed zero.
inline __m256 prepare_omod_f32(__m256 value, uint32_t mode) {
  auto magnitude = _mm256_and_si256(_mm256_castps_si256(value), _mm256_set1_epi32(INT32_MAX));
  auto tiny = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x00800000), magnitude);
  value = _mm256_andnot_ps(_mm256_castsi256_ps(tiny), value);
  if ((mode & GOC_ALU_OMOD_HALF) == GOC_ALU_OMOD_HALF) {
    auto underflow = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x01000000), magnitude);
    auto sign = _mm256_and_ps(value, _mm256_castsi256_ps(_mm256_set1_epi32(INT32_MIN)));
    value = _mm256_blendv_ps(value, sign, _mm256_castsi256_ps(underflow));
  }
  return value;
}

} // namespace goc

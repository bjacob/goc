// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {

// Clamp the wrapped sum of signed 32-bit lanes to the signed range.
inline __m256i saturate_signed_sum(__m256i x, __m256i y, __m256i sum) {
  auto overflow = _mm256_and_si256(_mm256_xor_si256(x, sum), _mm256_xor_si256(y, sum));
  auto limit = _mm256_xor_si256(_mm256_srai_epi32(x, 31), _mm256_set1_epi32(0x7fffffff));
  return _mm256_blendv_epi8(sum, limit, _mm256_srai_epi32(overflow, 31));
}

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
inline __m128 prepare_omod_f32(__m128 value, uint32_t mode) {
  auto magnitude = _mm_and_si128(_mm_castps_si128(value), _mm_set1_epi32(INT32_MAX));
  auto tiny = _mm_cmpgt_epi32(_mm_set1_epi32(0x00800000), magnitude);
  value = _mm_andnot_ps(_mm_castsi128_ps(tiny), value);
  if ((mode & GOC_ALU_OMOD_HALF) == GOC_ALU_OMOD_HALF) {
    auto underflow = _mm_cmpgt_epi32(_mm_set1_epi32(0x01000000), magnitude);
    auto sign = _mm_and_ps(value, _mm_castsi128_ps(_mm_set1_epi32(INT32_MIN)));
    value = _mm_blendv_ps(value, sign, _mm_castsi128_ps(underflow));
  }
  return value;
}

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

// Prepare FP64 arithmetic results for a nonzero OMOD.
inline __m256d prepare_omod_f64(__m256d value, uint32_t mode) {
  auto magnitude = _mm256_and_si256(_mm256_castpd_si256(value), _mm256_set1_epi64x(INT64_MAX));
  auto tiny = _mm256_cmpgt_epi64(_mm256_set1_epi64x(INT64_C(0x0010000000000000)), magnitude);
  value = _mm256_andnot_pd(_mm256_castsi256_pd(tiny), value);
  if ((mode & GOC_ALU_OMOD_HALF) == GOC_ALU_OMOD_HALF) {
    auto underflow = _mm256_cmpgt_epi64(_mm256_set1_epi64x(INT64_C(0x0020000000000000)), magnitude);
    auto sign = _mm256_and_pd(value, _mm256_castsi256_pd(_mm256_set1_epi64x(INT64_MIN)));
    value = _mm256_blendv_pd(value, sign, _mm256_castsi256_pd(underflow));
  }
  return value;
}

} // namespace goc

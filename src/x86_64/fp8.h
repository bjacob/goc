// SPDX-License-Identifier: MIT

#pragma once

#include <immintrin.h>

namespace goc {

// Widen the low byte of each 32-bit lane from OCP E4M3FN/E5M2 to FP32.
// Finite inputs widen exactly; NaNs become quiet NaNs, preserving the sign.
template <bool Bf8> inline __m256 widen_fp8(__m256i bytes) {
  constexpr int fraction_bits = Bf8 ? 2 : 3;
  constexpr int bias = Bf8 ? 15 : 7;
  auto magnitude = _mm256_and_si256(bytes, _mm256_set1_epi32(127));
  auto fraction = _mm256_and_si256(magnitude, _mm256_set1_epi32((1 << fraction_bits) - 1));
  auto normal = _mm256_add_epi32(_mm256_slli_epi32(magnitude, 23 - fraction_bits),
                                 _mm256_set1_epi32((127 - bias) << 23));
  auto subnormal =
      _mm256_mul_ps(_mm256_cvtepi32_ps(fraction), _mm256_set1_ps(Bf8 ? 0x1p-16f : 0x1p-9f));
  auto small = _mm256_cmpgt_epi32(_mm256_set1_epi32(1 << fraction_bits), magnitude);
  auto result = _mm256_blendv_epi8(normal, _mm256_castps_si256(subnormal), small);
  if constexpr (Bf8) {
    auto special = _mm256_cmpgt_epi32(magnitude, _mm256_set1_epi32(123));
    auto nan = _mm256_cmpgt_epi32(magnitude, _mm256_set1_epi32(124));
    auto exceptional = _mm256_or_si256(_mm256_set1_epi32(0x7f800000),
                                       _mm256_and_si256(nan, _mm256_set1_epi32(0x00400000)));
    result = _mm256_blendv_epi8(result, exceptional, special);
  } else {
    auto nan = _mm256_cmpeq_epi32(magnitude, _mm256_set1_epi32(127));
    result = _mm256_blendv_epi8(result, _mm256_set1_epi32(0x7fc00000), nan);
  }
  auto sign = _mm256_slli_epi32(_mm256_and_si256(bytes, _mm256_set1_epi32(128)), 24);
  return _mm256_castsi256_ps(_mm256_or_si256(result, sign));
}

} // namespace goc

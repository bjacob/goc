// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {

template <bool Bf16> __m256 half_input(__m256i words, int shift, uint32_t mode) {
  auto halves = _mm256_and_si256(_mm256_srl_epi32(words, _mm_cvtsi32_si128(shift)),
                                 _mm256_set1_epi32(mode & GOC_ALU_ABS_A ? 0x7fff : 0xffff));
  halves = _mm256_xor_si256(halves, _mm256_set1_epi32(mode & GOC_ALU_NEG_A ? 0x8000 : 0));
  if constexpr (Bf16) {
    auto zero_exp = _mm256_cmpeq_epi32(_mm256_and_si256(halves, _mm256_set1_epi32(0x7f80)),
                                       _mm256_setzero_si256());
    halves =
        _mm256_blendv_epi8(halves, _mm256_and_si256(halves, _mm256_set1_epi32(0x8000)), zero_exp);
    return _mm256_castsi256_ps(_mm256_slli_epi32(halves, 16));
  } else
    return _mm256_cvtph_ps(
        _mm_packus_epi32(_mm256_castsi256_si128(halves), _mm256_extracti128_si256(halves, 1)));
}

template <bool Bf16> __m256i half_narrow(__m256 value, bool saturate) {
  auto bits = _mm256_castps_si256(value);
  auto magnitude = _mm256_and_si256(bits, _mm256_set1_epi32(INT32_MAX));
  if constexpr (Bf16) {
    auto tie = _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1));
    auto result = _mm256_srli_epi32(
        _mm256_add_epi32(bits, _mm256_add_epi32(tie, _mm256_set1_epi32(0x7fff))), 16);
    auto nan = _mm256_cmpgt_epi32(magnitude, _mm256_set1_epi32(0x7f800000));
    result = _mm256_blendv_epi8(
        result, _mm256_or_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(0x40)), nan);
    auto zero_exp = _mm256_cmpeq_epi32(_mm256_and_si256(result, _mm256_set1_epi32(0x7f80)),
                                       _mm256_setzero_si256());
    return _mm256_blendv_epi8(result, _mm256_and_si256(result, _mm256_set1_epi32(0x8000)),
                              zero_exp);
  } else {
    auto result = _mm256_cvtepu16_epi32(
        _mm256_cvtps_ph(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    if (saturate) {
      auto finite = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x7f800000), magnitude);
      auto infinity = _mm256_cmpeq_epi32(_mm256_and_si256(result, _mm256_set1_epi32(0x7fff)),
                                         _mm256_set1_epi32(0x7c00));
      result = _mm256_add_epi32(result, _mm256_and_si256(finite, infinity));
    }
    return result;
  }
}

} // namespace goc

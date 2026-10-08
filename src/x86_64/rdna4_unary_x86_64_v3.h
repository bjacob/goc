// SPDX-License-Identifier: MIT

#pragma once

#include "rdna4_unary.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {

// Half inputs have already widened exactly to normal FP32 values (or zero).
template <Unary Op, bool Half = false> __m256 unary_value(__m256 value) {
  static_assert(Op != Unary::Exp && Op != Unary::Log);
  if constexpr (Op == Unary::Fract) {
    value = _mm256_sub_ps(value, _mm256_floor_ps(value));
    auto limit = _mm256_castsi256_ps(_mm256_set1_epi32(Half ? 0x3f7fe000 : 0x3f7fffff));
    value = _mm256_blendv_ps(value, limit, _mm256_cmp_ps(value, limit, _CMP_GT_OQ));
  }
  if constexpr (Op == Unary::FrexpMant) {
    auto original = _mm256_castps_si256(value);
    auto magnitude = _mm256_and_si256(original, _mm256_set1_epi32(INT32_MAX));
    auto normalized = value;
    if constexpr (!Half) {
      auto subnormal = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x00800000), magnitude);
      normalized = _mm256_blendv_ps(value, _mm256_mul_ps(value, _mm256_set1_ps(0x1p24f)),
                                    _mm256_castsi256_ps(subnormal));
    }
    auto mantissa = _mm256_or_si256(
        _mm256_and_si256(_mm256_castps_si256(normalized), _mm256_set1_epi32(int(0x807fffff))),
        _mm256_set1_epi32(0x3f000000));
    auto special =
        _mm256_or_si256(_mm256_cmpeq_epi32(magnitude, _mm256_setzero_si256()),
                        _mm256_cmpgt_epi32(magnitude, _mm256_set1_epi32(0x7f800000 - 1)));
    value = _mm256_castsi256_ps(_mm256_blendv_epi8(mantissa, original, special));
  }
  if constexpr (Op == Unary::Trunc)
    value = _mm256_round_ps(value, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
  if constexpr (Op == Unary::Ceil)
    value = _mm256_round_ps(value, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
  if constexpr (Op == Unary::Floor)
    value = _mm256_round_ps(value, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
  if constexpr (Op == Unary::Rndne)
    value = _mm256_round_ps(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  if constexpr (Op == Unary::Sqrt || Op == Unary::Rsq)
    value = _mm256_sqrt_ps(value);
  if constexpr (Op == Unary::Rcp || Op == Unary::Rsq)
    value = _mm256_div_ps(_mm256_set1_ps(1), value);
  return value;
}

} // namespace goc

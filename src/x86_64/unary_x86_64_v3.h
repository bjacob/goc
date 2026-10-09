// SPDX-License-Identifier: MIT

#pragma once

#include "unary.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {

// Approximates exp2 for widened FP16 inputs. Finite clamping preserves every
// final FP16 overflow/underflow outcome, including OMOD and saturation.
inline __m256 half_exp2(__m256 input) {
  auto x = _mm256_min_ps(_mm256_max_ps(input, _mm256_set1_ps(-64)), _mm256_set1_ps(64));
  auto exponent = _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  auto r = _mm256_sub_ps(x, exponent);
  // Taylor coefficients (ln(2)^n / n!) on the reduced interval [-0.5, 0.5].
  auto p = _mm256_set1_ps(1.525273380405984e-5f);
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.0001540353039338161f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.001333355814642844f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.009618129107628477f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.05550410866482158f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.2402265069591007f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.6931471805599453f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1));
  auto scale = _mm256_castsi256_ps(_mm256_slli_epi32(
      _mm256_add_epi32(_mm256_cvttps_epi32(exponent), _mm256_set1_epi32(127)), 23));
  auto result = _mm256_mul_ps(p, scale);
  auto infinity = _mm256_castsi256_ps(_mm256_set1_epi32(0x7f800000));
  auto negative_infinity = _mm256_castsi256_ps(_mm256_set1_epi32(int(0xff800000u)));
  result = _mm256_blendv_ps(result, _mm256_setzero_ps(),
                            _mm256_cmp_ps(input, negative_infinity, _CMP_EQ_OQ));
  auto positive_inf_or_nan = _mm256_or_ps(_mm256_cmp_ps(input, infinity, _CMP_EQ_OQ),
                                          _mm256_cmp_ps(input, input, _CMP_UNORD_Q));
  return _mm256_blendv_ps(result, input, positive_inf_or_nan);
}

// Approximates log2 for widened FP16 inputs, including their subnormals (which
// are normal FP32 values). Zero returns -infinity; negative inputs return NaN.
inline __m256 half_log2(__m256 input) {
  auto bits = _mm256_castps_si256(input);
  auto magnitude = _mm256_and_si256(bits, _mm256_set1_epi32(0x7fffffff));
  auto exponent = _mm256_sub_epi32(_mm256_srli_epi32(magnitude, 23), _mm256_set1_epi32(127));
  auto m = _mm256_castsi256_ps(_mm256_or_si256(
      _mm256_and_si256(magnitude, _mm256_set1_epi32(0x007fffff)), _mm256_set1_epi32(0x3f800000)));
  auto upper = _mm256_cmp_ps(m, _mm256_set1_ps(1.4142135623730951f), _CMP_GT_OQ);
  m = _mm256_blendv_ps(m, _mm256_mul_ps(m, _mm256_set1_ps(0.5f)), upper);
  exponent = _mm256_sub_epi32(exponent, _mm256_castps_si256(upper));
  // log(m) = 2 * (z + z^3/3 + z^5/5 + ...), |z| <= 0.171573.
  auto z = _mm256_div_ps(_mm256_sub_ps(m, _mm256_set1_ps(1)), _mm256_add_ps(m, _mm256_set1_ps(1)));
  auto z2 = _mm256_mul_ps(z, z);
  auto p = _mm256_set1_ps(1.0f / 9);
  p = _mm256_fmadd_ps(p, z2, _mm256_set1_ps(1.0f / 7));
  p = _mm256_fmadd_ps(p, z2, _mm256_set1_ps(1.0f / 5));
  p = _mm256_fmadd_ps(p, z2, _mm256_set1_ps(1.0f / 3));
  p = _mm256_fmadd_ps(p, z2, _mm256_set1_ps(1));
  auto result = _mm256_fmadd_ps(_mm256_mul_ps(z, _mm256_set1_ps(2.885390081777927f)), p,
                                _mm256_cvtepi32_ps(exponent));
  auto infinity = _mm256_set1_epi32(0x7f800000);
  result =
      _mm256_blendv_ps(result, input, _mm256_castsi256_ps(_mm256_cmpeq_epi32(magnitude, infinity)));
  auto invalid =
      _mm256_or_si256(_mm256_srai_epi32(bits, 31), _mm256_cmpgt_epi32(magnitude, infinity));
  result = _mm256_blendv_ps(result, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fc00000)),
                            _mm256_castsi256_ps(invalid));
  return _mm256_blendv_ps(
      result, _mm256_castsi256_ps(_mm256_set1_epi32(int(0xff800000u))),
      _mm256_castsi256_ps(_mm256_cmpeq_epi32(magnitude, _mm256_setzero_si256())));
}

// Half inputs have already widened exactly to normal FP32 values (or zero).
template <Unary Op, bool Half = false> __m256 unary_value(__m256 value) {
  static_assert(Half || (Op != Unary::Exp && Op != Unary::Log));
  if constexpr (Op == Unary::Exp)
    return half_exp2(value);
  if constexpr (Op == Unary::Log)
    return half_log2(value);
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

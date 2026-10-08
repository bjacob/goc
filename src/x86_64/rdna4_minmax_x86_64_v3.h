// SPDX-License-Identifier: MIT

#pragma once

#include <immintrin.h>
#include <stdint.h>

namespace goc {

// Correct the x86 selection rules for NaNs and opposite-signed zeros.
template <bool Maximum, bool Propagate> __m256 minmax(__m256 x, __m256 y) {
  auto a = _mm256_castps_si256(x), b = _mm256_castps_si256(y);
  auto magnitude = _mm256_set1_epi32(INT32_MAX);
  auto infinity = _mm256_set1_epi32(0x7f800000);
  auto quiet = _mm256_set1_epi32(0x00400000);
  auto an = _mm256_cmpgt_epi32(_mm256_and_si256(a, magnitude), infinity);
  auto bn = _mm256_cmpgt_epi32(_mm256_and_si256(b, magnitude), infinity);
  auto value = Maximum ? _mm256_max_ps(x, y) : _mm256_min_ps(x, y);
  auto equal = _mm256_cmp_ps(x, y, _CMP_EQ_OQ);
  auto tie = Maximum ? _mm256_and_si256(a, b) : _mm256_or_si256(a, b);
  value = _mm256_blendv_ps(value, _mm256_castsi256_ps(tie), equal);
  if constexpr (Propagate) {
    auto snb = _mm256_andnot_si256(_mm256_cmpeq_epi32(_mm256_and_si256(b, quiet), quiet), bn);
    auto sna = _mm256_andnot_si256(_mm256_cmpeq_epi32(_mm256_and_si256(a, quiet), quiet), an);
    auto selected = _mm256_blendv_epi8(b, a, an);
    selected = _mm256_blendv_epi8(selected, b, snb);
    selected = _mm256_blendv_epi8(selected, a, sna);
    value = _mm256_blendv_ps(value, _mm256_castsi256_ps(_mm256_or_si256(selected, quiet)),
                             _mm256_castsi256_ps(_mm256_or_si256(an, bn)));
  } else {
    value = _mm256_blendv_ps(value, x, _mm256_castsi256_ps(bn));
    value = _mm256_blendv_ps(value, y, _mm256_castsi256_ps(an));
    value = _mm256_blendv_ps(value, _mm256_castsi256_ps(_mm256_or_si256(a, quiet)),
                             _mm256_castsi256_ps(_mm256_and_si256(an, bn)));
  }
  return value;
}

// Correct the x86 selection rules for NaNs and opposite-signed zeros.
template <bool Maximum, bool Propagate> __m256d minmax(__m256d x, __m256d y) {
  auto a = _mm256_castpd_si256(x), b = _mm256_castpd_si256(y);
  auto magnitude = _mm256_set1_epi64x(INT64_MAX);
  auto infinity = _mm256_set1_epi64x(INT64_C(0x7ff0000000000000));
  auto quiet = _mm256_set1_epi64x(INT64_C(0x0008000000000000));
  auto an = _mm256_cmpgt_epi64(_mm256_and_si256(a, magnitude), infinity);
  auto bn = _mm256_cmpgt_epi64(_mm256_and_si256(b, magnitude), infinity);
  auto value = Maximum ? _mm256_max_pd(x, y) : _mm256_min_pd(x, y);
  auto equal = _mm256_cmp_pd(x, y, _CMP_EQ_OQ);
  auto tie = Maximum ? _mm256_and_si256(a, b) : _mm256_or_si256(a, b);
  value = _mm256_blendv_pd(value, _mm256_castsi256_pd(tie), equal);
  if constexpr (Propagate) {
    auto snb = _mm256_andnot_si256(_mm256_cmpeq_epi64(_mm256_and_si256(b, quiet), quiet), bn);
    auto sna = _mm256_andnot_si256(_mm256_cmpeq_epi64(_mm256_and_si256(a, quiet), quiet), an);
    auto selected = _mm256_blendv_epi8(b, a, an);
    selected = _mm256_blendv_epi8(selected, b, snb);
    selected = _mm256_blendv_epi8(selected, a, sna);
    value = _mm256_blendv_pd(value, _mm256_castsi256_pd(_mm256_or_si256(selected, quiet)),
                             _mm256_castsi256_pd(_mm256_or_si256(an, bn)));
  } else {
    value = _mm256_blendv_pd(value, x, _mm256_castsi256_pd(bn));
    value = _mm256_blendv_pd(value, y, _mm256_castsi256_pd(an));
    value = _mm256_blendv_pd(value, _mm256_castsi256_pd(_mm256_or_si256(a, quiet)),
                             _mm256_castsi256_pd(_mm256_and_si256(an, bn)));
  }
  return value;
}

} // namespace goc

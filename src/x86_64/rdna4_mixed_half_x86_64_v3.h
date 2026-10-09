// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rdna4_mixed_fma_scalar.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {

// Round four finite FP64 values to FP16 encodings in four 32-bit elements.
template <bool Rtz> inline __m128i mixed_half_narrow4(__m256d value, bool saturate) {
  // Adapted from rocjitsu's round_finite_f16_simd, with nearest-even and round-toward-zero
  // variants.
  auto bits = _mm256_castpd_si256(value);
  auto exponent = _mm256_and_si256(_mm256_srli_epi64(bits, 52), _mm256_set1_epi64x(2047));
  auto significand = _mm256_or_si256(_mm256_and_si256(bits, _mm256_set1_epi64x(0x000fffffffffffff)),
                                     _mm256_set1_epi64x(INT64_C(1) << 52));
  auto normal = _mm256_cmpgt_epi64(exponent, _mm256_set1_epi64x(1008));
  auto not_tiny = _mm256_cmpgt_epi64(exponent, _mm256_set1_epi64x(997));
  auto shift = _mm256_blendv_epi8(_mm256_sub_epi64(_mm256_set1_epi64x(1051), exponent),
                                  _mm256_set1_epi64x(42), normal);
  shift = _mm256_blendv_epi8(_mm256_set1_epi64x(53), shift, not_tiny);
  auto rounded = _mm256_srlv_epi64(significand, shift);
  rounded = _mm256_add_epi64(
      rounded,
      _mm256_and_si256(
          normal, _mm256_slli_epi64(_mm256_sub_epi64(exponent, _mm256_set1_epi64x(1009)), 10)));
  if constexpr (!Rtz) {
    auto unit = _mm256_sllv_epi64(_mm256_set1_epi64x(1), shift);
    auto remainder = _mm256_and_si256(significand, _mm256_sub_epi64(unit, _mm256_set1_epi64x(1)));

    auto halfway = _mm256_srli_epi64(unit, 1);
    auto increment = _mm256_or_si256(
        _mm256_cmpgt_epi64(remainder, halfway),
        _mm256_and_si256(_mm256_cmpeq_epi64(remainder, halfway),
                         _mm256_sub_epi64(_mm256_setzero_si256(),
                                          _mm256_and_si256(rounded, _mm256_set1_epi64x(1)))));
    rounded = _mm256_sub_epi64(rounded, _mm256_and_si256(increment, not_tiny));
  }
  auto overflow = _mm256_or_si256(_mm256_cmpgt_epi64(exponent, _mm256_set1_epi64x(1038)),
                                  _mm256_cmpgt_epi64(rounded, _mm256_set1_epi64x(0x7bff)));
  rounded = _mm256_blendv_epi8(rounded, _mm256_set1_epi64x((saturate || Rtz) ? 0x7bff : 0x7c00),
                               overflow);
  auto sign = _mm256_and_si256(_mm256_srli_epi64(bits, 48), _mm256_set1_epi64x(0x8000));
  auto packed = _mm256_shuffle_epi32(_mm256_or_si256(sign, rounded), _MM_SHUFFLE(2, 0, 2, 0));
  return _mm_unpacklo_epi64(_mm256_castsi256_si128(packed), _mm256_extracti128_si256(packed, 1));
}

template <bool Rtz>
inline __m128i mixed_half_value4(__m128 a, __m128 b, __m128 c, bool saturate, bool clamp) {
  auto x = _mm256_cvtps_pd(a), y = _mm256_cvtps_pd(b), z = _mm256_cvtps_pd(c);
  auto product = _mm256_mul_pd(x, y);
  auto value = _mm256_add_pd(product, z);
  auto virtual_c = _mm256_sub_pd(value, product);
  auto error = _mm256_add_pd(_mm256_sub_pd(product, _mm256_sub_pd(value, virtual_c)),
                             _mm256_sub_pd(z, virtual_c));
  auto bits = _mm256_castpd_si256(value);
  auto even =
      _mm256_cmpeq_epi64(_mm256_and_si256(bits, _mm256_set1_epi64x(1)), _mm256_setzero_si256());
  auto adjust = _mm256_and_si256(
      even, _mm256_castpd_si256(_mm256_cmp_pd(error, _mm256_setzero_pd(), _CMP_NEQ_OQ)));
  auto same_sign =
      _mm256_cmpeq_epi64(_mm256_srli_epi64(_mm256_xor_si256(bits, _mm256_castpd_si256(error)), 63),
                         _mm256_setzero_si256());
  auto step = _mm256_blendv_epi8(_mm256_set1_epi64x(-1), _mm256_set1_epi64x(1), same_sign);
  bits = _mm256_add_epi64(bits, _mm256_and_si256(adjust, step));
  auto result = mixed_half_narrow4<Rtz>(_mm256_castsi256_pd(bits), saturate);

  auto aa = _mm_and_si128(_mm_castps_si128(a), _mm_set1_epi32(INT32_MAX));
  auto ab = _mm_and_si128(_mm_castps_si128(b), _mm_set1_epi32(INT32_MAX));
  auto ac = _mm_and_si128(_mm_castps_si128(c), _mm_set1_epi32(INT32_MAX));
  auto largest = _mm_max_epi32(_mm_max_epi32(aa, ab), ac);
  unsigned special = unsigned(
      _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpgt_epi32(largest, _mm_set1_epi32(0x7f7fffff)))));
  if (special) {
    uint32_t av[4], bv[4], cv[4], out[4];
    _mm_storeu_si128(reinterpret_cast<__m128i *>(av), _mm_castps_si128(a));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(bv), _mm_castps_si128(b));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(cv), _mm_castps_si128(c));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out), result);
    for (int lane = 0; lane < 4; ++lane)
      if ((special >> lane) & 1)
        out[lane] = goc::mixed_fma_special(av[lane], bv[lane], cv[lane]);
    result = _mm_loadu_si128(reinterpret_cast<const __m128i *>(out));
  }
  if (clamp) {
    auto negative = _mm_cmpgt_epi32(result, _mm_set1_epi32(0x7fff));
    auto nan =
        _mm_cmpgt_epi32(_mm_and_si128(result, _mm_set1_epi32(0x7fff)), _mm_set1_epi32(0x7c00));
    result = _mm_andnot_si128(_mm_or_si128(negative, nan),
                              _mm_min_epi32(result, _mm_set1_epi32(0x3c00)));
  }
  return result;
}

} // namespace goc

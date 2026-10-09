// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_mixed_fma.h"
#include "rdna4_mixed_fma_scalar.h"
#include "x86_64/rdna4_half_x86_64_v3.h"

#include <immintrin.h>
#include <stdint.h>

namespace {

__m256 input8(const uint32_t *p, uint32_t mode) {
  auto words = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p));
  if (mode & GOC_MIX_F16_A)
    return goc::half_input<false>(words, mode & GOC_ALU_HIGH_A ? 16 : 0, mode);
  return _mm256_castsi256_ps(_mm256_xor_si256(
      _mm256_and_si256(words, _mm256_set1_epi32(mode & GOC_ALU_ABS_A ? INT32_MAX : -1)),
      _mm256_set1_epi32(mode & GOC_ALU_NEG_A ? INT32_MIN : 0)));
}

__m128 input4(const uint32_t *p, uint32_t mode) {
  auto words = _mm_loadu_si128(reinterpret_cast<const __m128i *>(p));
  if (mode & GOC_MIX_F16_A) {
    auto halves =
        _mm_and_si128(_mm_srl_epi32(words, _mm_cvtsi32_si128(mode & GOC_ALU_HIGH_A ? 16 : 0)),
                      _mm_set1_epi32(65535));
    words = _mm_castps_si128(_mm_cvtph_ps(_mm_packus_epi32(halves, halves)));
  }
  return _mm_castsi128_ps(
      _mm_xor_si128(_mm_and_si128(words, _mm_set1_epi32(mode & GOC_ALU_ABS_A ? INT32_MAX : -1)),
                    _mm_set1_epi32(mode & GOC_ALU_NEG_A ? INT32_MIN : 0)));
}

// Round four finite FP64 values to FP16 encodings in four 32-bit elements.
__m128i narrow4(__m256d value, bool saturate) {
  // Adapted from rocjitsu's round_finite_f16_simd, restricted to nearest-even.
  auto bits = _mm256_castpd_si256(value);
  auto exponent = _mm256_and_si256(_mm256_srli_epi64(bits, 52), _mm256_set1_epi64x(2047));
  auto significand = _mm256_or_si256(_mm256_and_si256(bits, _mm256_set1_epi64x(0x000fffffffffffff)),
                                     _mm256_set1_epi64x(INT64_C(1) << 52));
  auto normal = _mm256_cmpgt_epi64(exponent, _mm256_set1_epi64x(1008));
  auto not_tiny = _mm256_cmpgt_epi64(exponent, _mm256_set1_epi64x(997));
  auto shift = _mm256_blendv_epi8(_mm256_sub_epi64(_mm256_set1_epi64x(1051), exponent),
                                  _mm256_set1_epi64x(42), normal);
  shift = _mm256_blendv_epi8(_mm256_set1_epi64x(53), shift, not_tiny);
  auto unit = _mm256_sllv_epi64(_mm256_set1_epi64x(1), shift);
  auto remainder = _mm256_and_si256(significand, _mm256_sub_epi64(unit, _mm256_set1_epi64x(1)));
  auto rounded = _mm256_srlv_epi64(significand, shift);
  rounded = _mm256_add_epi64(
      rounded,
      _mm256_and_si256(
          normal, _mm256_slli_epi64(_mm256_sub_epi64(exponent, _mm256_set1_epi64x(1009)), 10)));
  auto halfway = _mm256_srli_epi64(unit, 1);
  auto increment = _mm256_or_si256(
      _mm256_cmpgt_epi64(remainder, halfway),
      _mm256_and_si256(_mm256_cmpeq_epi64(remainder, halfway),
                       _mm256_sub_epi64(_mm256_setzero_si256(),
                                        _mm256_and_si256(rounded, _mm256_set1_epi64x(1)))));
  rounded = _mm256_sub_epi64(rounded, _mm256_and_si256(increment, not_tiny));
  auto overflow = _mm256_or_si256(_mm256_cmpgt_epi64(exponent, _mm256_set1_epi64x(1038)),
                                  _mm256_cmpgt_epi64(rounded, _mm256_set1_epi64x(0x7bff)));
  rounded = _mm256_blendv_epi8(rounded, _mm256_set1_epi64x(saturate ? 0x7bff : 0x7c00), overflow);
  auto sign = _mm256_and_si256(_mm256_srli_epi64(bits, 48), _mm256_set1_epi64x(0x8000));
  auto packed = _mm256_shuffle_epi32(_mm256_or_si256(sign, rounded), _MM_SHUFFLE(2, 0, 2, 0));
  return _mm_unpacklo_epi64(_mm256_castsi256_si128(packed), _mm256_extracti128_si256(packed, 1));
}

__m128i half_value4(__m128 a, __m128 b, __m128 c, bool saturate, bool clamp) {
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
  auto result = narrow4(_mm256_castsi256_pd(bits), saturate);

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

} // namespace

namespace goc {

template <MixedFma Dst>
void mixed_fma_x86_64_v3(bool saturate, uint32_t mask, uint32_t mode, uint32_t *d,
                         const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  if constexpr (Dst == MixedFma::Float) {
    for (int lane = 0; lane < 32; lane += 8) {
      auto x = input8(a + lane, mode), y = input8(b + lane, mode >> 1),
           z = input8(c + lane, mode >> 2);
      auto result = _mm256_fmadd_ps(x, y, z);
      if (mode & GOC_ALU_CLAMP)
        result = _mm256_min_ps(_mm256_max_ps(result, _mm256_setzero_ps()), _mm256_set1_ps(1));
      auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(mask >> lane)),
                                      _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
      _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), active,
                             _mm256_castps_si256(result));
    }
  } else {
    for (int lane = 0; lane < 32; lane += 4) {
      auto x = input4(a + lane, mode), y = input4(b + lane, mode >> 1),
           z = input4(c + lane, mode >> 2);
      auto result = half_value4(x, y, z, saturate, mode & GOC_ALU_CLAMP);
      constexpr unsigned shift = Dst == MixedFma::High ? 16 : 0;
      auto original = _mm_loadu_si128(reinterpret_cast<const __m128i *>(d + lane));
      auto output_mask = _mm_set1_epi32(int(uint32_t(65535) << shift));
      result = _mm_or_si128(_mm_slli_epi32(result, shift), _mm_andnot_si128(output_mask, original));
      auto active =
          _mm_sllv_epi32(_mm_set1_epi32(int(mask >> lane)), _mm_setr_epi32(31, 30, 29, 28));
      _mm_maskstore_epi32(reinterpret_cast<int *>(d + lane), active, result);
    }
  }
}

template void mixed_fma_x86_64_v3<MixedFma::Float>(bool, uint32_t, uint32_t, uint32_t *,
                                                   const uint32_t *, const uint32_t *,
                                                   const uint32_t *);
template void mixed_fma_x86_64_v3<MixedFma::Low>(bool, uint32_t, uint32_t, uint32_t *,
                                                 const uint32_t *, const uint32_t *,
                                                 const uint32_t *);
template void mixed_fma_x86_64_v3<MixedFma::High>(bool, uint32_t, uint32_t, uint32_t *,
                                                  const uint32_t *, const uint32_t *,
                                                  const uint32_t *);

} // namespace goc

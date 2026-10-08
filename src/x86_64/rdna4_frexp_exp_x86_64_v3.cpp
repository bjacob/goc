// SPDX-License-Identifier: MIT

#include "rdna4_frexp_exp.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {
namespace {

template <bool Fp64> void run(uint32_t mask, uint32_t *d, const uint32_t *const *a) {
  uint32_t result[32];
  for (int lane = 0; lane < 32; lane += Fp64 ? 4 : 8) {
    if constexpr (Fp64) {
      auto low =
          _mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i *>(a[0] + lane)));
      auto high =
          _mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i *>(a[1] + lane)));
      auto magnitude = _mm256_and_si256(_mm256_or_si256(low, _mm256_slli_epi64(high, 32)),
                                        _mm256_set1_epi64x(INT64_MAX));
      auto subnormal =
          _mm256_cmpgt_epi64(_mm256_set1_epi64x(INT64_C(0x0010000000000000)), magnitude);
      // Scale only subnormal magnitudes, avoiding overflow on large normals.
      auto small = _mm256_and_si256(magnitude, subnormal);
      auto scaled =
          _mm256_castpd_si256(_mm256_mul_pd(_mm256_castsi256_pd(small), _mm256_set1_pd(0x1p54)));
      auto normalized = _mm256_blendv_epi8(magnitude, scaled, subnormal);
      auto exponent = _mm256_sub_epi64(_mm256_srli_epi64(normalized, 52), _mm256_set1_epi64x(1022));
      exponent = _mm256_sub_epi64(exponent, _mm256_and_si256(subnormal, _mm256_set1_epi64x(54)));
      auto special = _mm256_or_si256(
          _mm256_cmpeq_epi64(magnitude, _mm256_setzero_si256()),
          _mm256_cmpgt_epi64(magnitude, _mm256_set1_epi64x(INT64_C(0x7ff0000000000000) - 1)));
      exponent = _mm256_andnot_si256(special, exponent);
      auto packed =
          _mm256_permutevar8x32_epi32(exponent, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
      _mm_storeu_si128(reinterpret_cast<__m128i *>(result + lane), _mm256_castsi256_si128(packed));
    } else {
      auto magnitude =
          _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a[0] + lane)),
                           _mm256_set1_epi32(INT32_MAX));
      auto subnormal = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x00800000), magnitude);
      auto small = _mm256_and_si256(magnitude, subnormal);
      auto scaled =
          _mm256_castps_si256(_mm256_mul_ps(_mm256_castsi256_ps(small), _mm256_set1_ps(0x1p24f)));
      auto normalized = _mm256_blendv_epi8(magnitude, scaled, subnormal);
      auto exponent = _mm256_sub_epi32(_mm256_srli_epi32(normalized, 23), _mm256_set1_epi32(126));
      exponent = _mm256_sub_epi32(exponent, _mm256_and_si256(subnormal, _mm256_set1_epi32(24)));
      auto special =
          _mm256_or_si256(_mm256_cmpeq_epi32(magnitude, _mm256_setzero_si256()),
                          _mm256_cmpgt_epi32(magnitude, _mm256_set1_epi32(0x7f800000 - 1)));
      exponent = _mm256_andnot_si256(special, exponent);
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(result + lane), exponent);
    }
  }
  for (int lane = 0; lane < 32; lane += 8) {
    auto lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(mask >> lane)),
                                    _mm256_sub_epi32(_mm256_set1_epi32(31), lanes));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), active,
                           _mm256_loadu_si256(reinterpret_cast<const __m256i *>(result + lane)));
  }
}

} // namespace

void frexp_exp_x86_64_v3(bool fp64, uint32_t mask, uint32_t *d, const uint32_t *const *a) {
  if (fp64)
    run<true>(mask, d, a);
  else
    run<false>(mask, d, a);
}

} // namespace goc

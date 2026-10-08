// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_ldexp.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {
namespace {

__m256i clamp64(__m256i value, int low, int high) {
  auto lo = _mm256_set1_epi64x(low), hi = _mm256_set1_epi64x(high);
  value = _mm256_blendv_epi8(value, lo, _mm256_cmpgt_epi64(lo, value));
  return _mm256_blendv_epi8(value, hi, _mm256_cmpgt_epi64(value, hi));
}

void run32(uint32_t mode, uint32_t result[2][32], const uint32_t *const *a, const uint32_t *b) {
  auto keep = _mm256_set1_epi32(mode & GOC_ALU_ABS_A ? INT32_MAX : -1);
  auto flip = _mm256_set1_epi32(mode & GOC_ALU_NEG_A ? INT32_MIN : 0);
  const double scales[] = {1, 2, 4, 0.5};
  for (int lane = 0; lane < 32; lane += 8) {
    auto raw = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a[0] + lane));
    auto adjustment = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
    adjustment = _mm256_min_epi32(_mm256_max_epi32(adjustment, _mm256_set1_epi32(-4096)),
                                  _mm256_set1_epi32(4096));
    raw = _mm256_xor_si256(_mm256_and_si256(raw, keep), flip);
    auto magnitude = _mm256_and_si256(raw, _mm256_set1_epi32(INT32_MAX));
    auto subnormal = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x800000), magnitude);
    auto small = _mm256_and_si256(raw, subnormal);
    auto scaled =
        _mm256_castps_si256(_mm256_mul_ps(_mm256_castsi256_ps(small), _mm256_set1_ps(0x1p24)));
    auto normalized = _mm256_blendv_epi8(raw, scaled, subnormal);
    auto exponent =
        _mm256_srli_epi32(_mm256_and_si256(normalized, _mm256_set1_epi32(INT32_MAX)), 23);
    exponent = _mm256_add_epi32(exponent, adjustment);
    exponent = _mm256_sub_epi32(exponent, _mm256_and_si256(subnormal, _mm256_set1_epi32(24)));
    auto significand = _mm256_and_si256(normalized, _mm256_set1_epi32((INT32_MIN | 0x7fffff)));
    auto normal = _mm256_or_si256(
        significand,
        _mm256_slli_epi32(_mm256_min_epi32(_mm256_max_epi32(exponent, _mm256_set1_epi32(1)),
                                           _mm256_set1_epi32(254)),
                          23));
    // Move an underflowing value into the normal range, then round once on
    // scaling back down. Clamping the integer exponent prevents wraparound.
    auto raised = _mm256_add_epi32(exponent, _mm256_set1_epi32(24));
    auto tiny = _mm256_or_si256(
        significand,
        _mm256_slli_epi32(_mm256_min_epi32(_mm256_max_epi32(raised, _mm256_set1_epi32(1)),
                                           _mm256_set1_epi32(254)),
                          23));
    auto underflow =
        _mm256_castps_si256(_mm256_mul_ps(_mm256_castsi256_ps(tiny), _mm256_set1_ps(0x1p-24)));
    auto bits =
        _mm256_blendv_epi8(normal, underflow, _mm256_cmpgt_epi32(_mm256_set1_epi32(1), exponent));
    auto sign = _mm256_and_si256(raw, _mm256_set1_epi32(INT32_MIN));
    bits = _mm256_blendv_epi8(bits, sign, _mm256_cmpgt_epi32(_mm256_set1_epi32(-23), exponent));
    bits = _mm256_blendv_epi8(bits, _mm256_or_si256(sign, _mm256_set1_epi32(0x7f800000)),
                              _mm256_cmpgt_epi32(exponent, _mm256_set1_epi32(254)));
    auto special =
        _mm256_or_si256(_mm256_cmpeq_epi32(magnitude, _mm256_setzero_si256()),
                        _mm256_cmpgt_epi32(magnitude, _mm256_set1_epi32(0x7f800000 - 1)));
    auto quiet = _mm256_and_si256(_mm256_cmpgt_epi32(magnitude, _mm256_set1_epi32(0x7f800000)),
                                  _mm256_set1_epi32(0x400000));
    bits = _mm256_blendv_epi8(bits, _mm256_or_si256(raw, quiet), special);
    auto value = _mm256_castsi256_ps(bits);
    if (mode & GOC_ALU_OMOD_HALF)
      value = _mm256_mul_ps(value, _mm256_set1_ps(scales[(mode >> 6) & 3]));
    if (mode & GOC_ALU_CLAMP)
      value = _mm256_min_ps(_mm256_max_ps(value, _mm256_setzero_ps()), _mm256_set1_ps(1));
    bits = _mm256_castps_si256(value);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[0] + lane), bits);
  }
}

void run64(uint32_t mode, uint32_t result[2][32], const uint32_t *const *a, const uint32_t *b) {
  auto keep = _mm256_set1_epi64x(mode & GOC_ALU_ABS_A ? INT64_MAX : -1);
  auto flip = _mm256_set1_epi64x(mode & GOC_ALU_NEG_A ? INT64_MIN : 0);
  const double scales[] = {1, 2, 4, 0.5};
  for (int lane = 0; lane < 32; lane += 4) {
    auto low =
        _mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i *>(a[0] + lane)));
    auto high =
        _mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i *>(a[1] + lane)));
    auto raw = _mm256_or_si256(low, _mm256_slli_epi64(high, 32));
    auto power = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + lane));
    power = _mm_min_epi32(_mm_max_epi32(power, _mm_set1_epi32(-4096)), _mm_set1_epi32(4096));
    auto adjustment = _mm256_cvtepi32_epi64(power);
    raw = _mm256_xor_si256(_mm256_and_si256(raw, keep), flip);
    auto magnitude = _mm256_and_si256(raw, _mm256_set1_epi64x(INT64_MAX));
    auto subnormal = _mm256_cmpgt_epi64(_mm256_set1_epi64x(INT64_C(0x10000000000000)), magnitude);
    auto small = _mm256_and_si256(raw, subnormal);
    auto scaled =
        _mm256_castpd_si256(_mm256_mul_pd(_mm256_castsi256_pd(small), _mm256_set1_pd(0x1p54)));
    auto normalized = _mm256_blendv_epi8(raw, scaled, subnormal);
    auto exponent =
        _mm256_srli_epi64(_mm256_and_si256(normalized, _mm256_set1_epi64x(INT64_MAX)), 52);
    exponent = _mm256_add_epi64(exponent, adjustment);
    exponent = _mm256_sub_epi64(exponent, _mm256_and_si256(subnormal, _mm256_set1_epi64x(54)));
    auto significand =
        _mm256_and_si256(normalized, _mm256_set1_epi64x((INT64_MIN | INT64_C(0xfffffffffffff))));
    auto normal = _mm256_or_si256(significand, _mm256_slli_epi64(clamp64(exponent, 1, 2046), 52));
    // Move an underflowing value into the normal range, then round once on
    // scaling back down. Clamping the integer exponent prevents wraparound.
    auto raised = _mm256_add_epi64(exponent, _mm256_set1_epi64x(54));
    auto tiny = _mm256_or_si256(significand, _mm256_slli_epi64(clamp64(raised, 1, 2046), 52));
    auto underflow =
        _mm256_castpd_si256(_mm256_mul_pd(_mm256_castsi256_pd(tiny), _mm256_set1_pd(0x1p-54)));
    auto bits =
        _mm256_blendv_epi8(normal, underflow, _mm256_cmpgt_epi64(_mm256_set1_epi64x(1), exponent));
    auto sign = _mm256_and_si256(raw, _mm256_set1_epi64x(INT64_MIN));
    bits = _mm256_blendv_epi8(bits, sign, _mm256_cmpgt_epi64(_mm256_set1_epi64x(-52), exponent));
    bits = _mm256_blendv_epi8(
        bits, _mm256_or_si256(sign, _mm256_set1_epi64x(INT64_C(0x7ff0000000000000))),
        _mm256_cmpgt_epi64(exponent, _mm256_set1_epi64x(2046)));
    auto special = _mm256_or_si256(
        _mm256_cmpeq_epi64(magnitude, _mm256_setzero_si256()),
        _mm256_cmpgt_epi64(magnitude, _mm256_set1_epi64x(INT64_C(0x7ff0000000000000) - 1)));
    auto quiet = _mm256_and_si256(
        _mm256_cmpgt_epi64(magnitude, _mm256_set1_epi64x(INT64_C(0x7ff0000000000000))),
        _mm256_set1_epi64x(INT64_C(0x8000000000000)));
    bits = _mm256_blendv_epi8(bits, _mm256_or_si256(raw, quiet), special);
    auto value = _mm256_castsi256_pd(bits);
    if (mode & GOC_ALU_OMOD_HALF)
      value = _mm256_mul_pd(value, _mm256_set1_pd(scales[(mode >> 6) & 3]));
    if (mode & GOC_ALU_CLAMP)
      value = _mm256_min_pd(_mm256_max_pd(value, _mm256_setzero_pd()), _mm256_set1_pd(1));
    bits = _mm256_castpd_si256(value);
    auto packed = _mm256_permutevar8x32_epi32(bits, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(result[0] + lane), _mm256_castsi256_si128(packed));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(result[1] + lane),
                     _mm256_extracti128_si256(packed, 1));
  }
}

} // namespace

void ldexp_x86_64_v3(bool fp64, uint32_t mask, uint32_t mode, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *b) {
  uint32_t result[2][32];
  if (fp64)
    run64(mode, result, a, b);
  else
    run32(mode, result, a, b);
  for (int reg = 0; reg < (fp64 ? 2 : 1); ++reg)
    for (int lane = 0; lane < 32; lane += 8) {
      auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(mask >> lane)),
                                      _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
      _mm256_maskstore_epi32(
          reinterpret_cast<int *>(d[reg] + lane), active,
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(result[reg] + lane)));
    }
}

} // namespace goc

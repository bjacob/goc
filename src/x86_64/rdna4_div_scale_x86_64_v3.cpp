// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Vectorized form of the rocjitsu-derived division pre-scaling model.

#include "goc/goc.h"
#include "rdna4_div_scale.h"
#include "rdna4_division.h"
#include "x86_64/rdna4_division_x86_64_v3.h"

#include <immintrin.h>
#include <stdint.h>

namespace {

template <bool Wide> using Ops = goc::DivisionOpsV3<Wide>;

template <unsigned Width>
__m256i scale_value(__m256i a, __m256i b, __m256i c, uint32_t mode,
                    typename Ops<Width == 64>::Mask &post) {
  using F = goc::DivisionFormat<Width>;
  using O = Ops<Width == 64>;
  constexpr int threshold = Width == 32 ? 96 : 768, scale = Width == 32 ? 64 : 128;
  auto zero = O::set(0), one = O::set(1), inf = O::set(F::infinity), signbit = O::set(F::sign),
       magmask = O::set(F::sign - 1);
  auto mag = _mm256_and_si256(a, magmask), bm = _mm256_and_si256(b, magmask),
       cm = _mm256_and_si256(c, magmask);
  auto de = O::shr(_mm256_and_si256(b, inf), F::fraction),
       ne = O::shr(_mm256_and_si256(c, inf), F::fraction);
  auto delta = O::sub(ne, de);
  auto denominator = O::eq(a, b);
  auto large_delta = O::gt(delta, O::set(threshold - 1)),
       small_delta = O::gt(O::set(uint64_t(-int64_t(threshold) + 1)), delta);
  post = O::either(large_delta, small_delta);
  auto adjustment = O::select(O::gt(O::set(F::fraction + 2), ne), O::set(scale), zero);
  adjustment = O::select(small_delta, O::select(denominator, zero, O::set(scale)), adjustment);
  auto high_denominator = O::gt(de, O::set(2 * F::bias - 2));
  adjustment = O::select(high_denominator,
                         O::select(O::both(small_delta, O::inverse(denominator)), zero,
                                   O::set(uint64_t(-int64_t(scale)))),
                         adjustment);
  adjustment = O::select(O::eq(de, zero), O::set(scale), adjustment);
  adjustment = O::select(large_delta, O::select(denominator, O::set(scale), zero), adjustment);
  auto fraction_mask = O::set((uint64_t(1) << F::fraction) - 1),
       hidden = O::set(uint64_t(1) << F::fraction);
  auto fraction = _mm256_and_si256(a, fraction_mask);
  // Encoding 2^fraction + significand then subtracting 2^fraction converts
  // the integer significand exactly, without touching subnormal FP operands.
  // The result supplies a normalized mantissa and its leading-bit exponent.
  auto magic = O::set(uint64_t(F::bias + F::fraction) << F::fraction);
  auto encoded = _mm256_or_si256(magic, fraction);
  __m256i normalized;
  if constexpr (Width == 64)
    normalized =
        _mm256_castpd_si256(_mm256_sub_pd(_mm256_castsi256_pd(encoded), _mm256_set1_pd(0x1p52)));
  else
    normalized =
        _mm256_castps_si256(_mm256_sub_ps(_mm256_castsi256_ps(encoded), _mm256_set1_ps(0x1p23f)));
  auto original_exponent = O::shr(_mm256_and_si256(a, inf), F::fraction);
  auto subnormal = O::eq(original_exponent, zero);
  auto small_exponent = O::sub(O::shr(_mm256_and_si256(normalized, inf), F::fraction),
                               O::set(F::bias + F::fraction - 1));
  auto exponent = O::add(O::select(subnormal, small_exponent, original_exponent), adjustment);
  auto mantissa = _mm256_and_si256(O::select(subnormal, normalized, a), fraction_mask);
  auto sign = _mm256_and_si256(a, signbit);
  auto result = _mm256_or_si256(sign, _mm256_or_si256(O::shl(exponent, F::fraction), mantissa));
  auto shift = O::sub(one, exponent);
  shift = O::select(O::gt(one, shift), one, shift);
  shift = O::select(O::gt(shift, O::set(F::fraction + 2)), O::set(F::fraction + 2), shift);
  auto significand = _mm256_or_si256(mantissa, hidden);
  __m256i half, kept, rounded;
  if constexpr (Width == 64) {
    half = _mm256_sllv_epi64(one, O::sub(shift, one));
    kept = _mm256_srlv_epi64(significand, shift);
    rounded = _mm256_srlv_epi64(
        O::add(significand, O::add(O::sub(half, one), _mm256_and_si256(kept, one))), shift);
  } else {
    half = _mm256_sllv_epi32(one, O::sub(shift, one));
    kept = _mm256_srlv_epi32(significand, shift);
    rounded = _mm256_srlv_epi32(
        O::add(significand, O::add(O::sub(half, one), _mm256_and_si256(kept, one))), shift);
  }
  result = O::select(O::gt(one, exponent), _mm256_or_si256(sign, rounded), result);
  result = O::select(O::gt(exponent, O::set(2 * F::bias)), _mm256_or_si256(sign, inf), result);
  auto quiet = O::select(O::gt(mag, inf), O::set(F::quiet), zero);
  result = O::select(O::either(O::eq(mag, zero), O::inverse(O::gt(inf, mag))),
                     _mm256_or_si256(a, quiet), result);
  result = O::select(O::either(O::eq(bm, zero), O::eq(cm, zero)),
                     O::set(F::sign | F::infinity | F::quiet), result);
  auto overflow = inf;
  unsigned omod = (mode >> 6) & 3;
  if (omod) {
    auto exponent = _mm256_and_si256(result, inf), output_sign = _mm256_and_si256(result, signbit),
         step = O::set(uint64_t(1) << F::fraction);
    __m256i scaled;
    if (omod == 3)
      scaled = O::select(O::eq(exponent, step), output_sign, O::sub(result, step));
    else {
      auto increment = O::set(uint64_t(omod) << F::fraction);
      scaled = O::select(O::inverse(O::gt(O::sub(inf, increment), exponent)),
                         _mm256_or_si256(output_sign, overflow), O::add(result, increment));
    }
    result = O::select(O::eq(exponent, inf), result, scaled);
    result = O::select(O::eq(exponent, zero), zero, result);
  }
  if (mode & GOC_ALU_CLAMP) {
    auto negative = O::inverse(O::eq(_mm256_and_si256(result, signbit), zero));
    result = O::select(O::either(negative, O::gt(result, inf)), zero, result);
    auto one = O::set(uint64_t(F::bias) << F::fraction);
    result = O::select(O::gt(result, one), one, result);
  }
  return result;
}

template <unsigned Width>
__m256i load(const uint32_t *const *v, unsigned lane, uint32_t mode, unsigned operand) {
  using F = goc::DivisionFormat<Width>;
  using O = Ops<Width == 64>;
  __m256i result;
  if constexpr (Width == 64) {
    auto low =
        _mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i *>(v[0] + lane)));
    auto high =
        _mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i *>(v[1] + lane)));
    result = _mm256_or_si256(low, _mm256_slli_epi64(high, 32));
  } else {
    result = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v[0] + lane));
  }
  auto flip = O::set(mode & (GOC_ALU_NEG_A << operand) ? F::sign : 0);
  return _mm256_xor_si256(result, flip);
}

} // namespace

namespace goc {

template <unsigned Width>
uint32_t div_scale_x86_64_v3(uint32_t mask, uint32_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c) {
  constexpr unsigned lanes = Width == 64 ? 4 : 8;
  uint32_t staged[2][32], conditions = 0;
  for (unsigned lane = 0; lane < 32; lane += lanes) {
    typename Ops<Width == 64>::Mask post;
    auto result = scale_value<Width>(load<Width>(a, lane, mode, 0), load<Width>(b, lane, mode, 1),
                                     load<Width>(c, lane, mode, 2), mode, post);
    if constexpr (Width == 64)
      conditions |= uint32_t(_mm256_movemask_pd(_mm256_castsi256_pd(post))) << lane;
    else
      conditions |= uint32_t(_mm256_movemask_ps(_mm256_castsi256_ps(post))) << lane;
    if constexpr (Width == 64) {
      auto words = _mm256_permutevar8x32_epi32(result, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
      _mm_storeu_si128(reinterpret_cast<__m128i *>(staged[0] + lane),
                       _mm256_castsi256_si128(words));
      _mm_storeu_si128(reinterpret_cast<__m128i *>(staged[1] + lane),
                       _mm256_extracti128_si256(words, 1));
    } else {
      auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(mask >> lane)),
                                      _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
      _mm256_maskstore_epi32(reinterpret_cast<int *>(d[0] + lane), active, result);
    }
  }
  // Preserve cross-half aliases by committing D0 then D1 after all reads.
  if constexpr (Width == 64)
    for (unsigned reg = 0; reg < 2; ++reg)
      for (unsigned lane = 0; lane < 32; lane += 8) {
        auto result = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(staged[reg] + lane));
        auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(mask >> lane)),
                                        _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
        _mm256_maskstore_epi32(reinterpret_cast<int *>(d[reg] + lane), active, result);
      }
  return conditions & mask;
}

template uint32_t div_scale_x86_64_v3<32>(uint32_t, uint32_t, uint32_t *const *,
                                          const uint32_t *const *, const uint32_t *const *,
                                          const uint32_t *const *);
template uint32_t div_scale_x86_64_v3<64>(uint32_t, uint32_t, uint32_t *const *,
                                          const uint32_t *const *, const uint32_t *const *,
                                          const uint32_t *const *);

} // namespace goc

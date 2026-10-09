// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Vectorized form of the rocjitsu-derived division fixup model.

#include "goc/goc.h"
#include "rdna4_div_fixup.h"

#include <immintrin.h>
#include <stdint.h>

namespace {

template <bool Wide> struct Ops {
  using V = __m256i;
  using Mask = __m256i;

  static V set(uint64_t x) {
    if constexpr (Wide)
      return _mm256_set1_epi64x(int64_t(x));
    else
      return _mm256_set1_epi32(int32_t(x));
  }

  static V add(V a, V b) {
    if constexpr (Wide)
      return _mm256_add_epi64(a, b);
    else
      return _mm256_add_epi32(a, b);
  }

  static V sub(V a, V b) {
    if constexpr (Wide)
      return _mm256_sub_epi64(a, b);
    else
      return _mm256_sub_epi32(a, b);
  }

  static V shr(V a, int count) {
    if constexpr (Wide)
      return _mm256_srli_epi64(a, count);
    else
      return _mm256_srli_epi32(a, count);
  }

  static V shl(V a, int count) {
    if constexpr (Wide)
      return _mm256_slli_epi64(a, count);
    else
      return _mm256_slli_epi32(a, count);
  }

  static Mask gt(V a, V b) {
    if constexpr (Wide)
      return _mm256_cmpgt_epi64(a, b);
    else
      return _mm256_cmpgt_epi32(a, b);
  }

  static Mask eq(V a, V b) {
    if constexpr (Wide)
      return _mm256_cmpeq_epi64(a, b);
    else
      return _mm256_cmpeq_epi32(a, b);
  }

  static Mask both(Mask a, Mask b) { return _mm256_and_si256(a, b); }

  static Mask either(Mask a, Mask b) { return _mm256_or_si256(a, b); }

  static Mask inverse(Mask a) { return _mm256_xor_si256(a, _mm256_set1_epi32(-1)); }

  static V select(Mask m, V yes, V no) { return _mm256_blendv_epi8(no, yes, m); }
};

template <unsigned Width>
__m256i value(__m256i a, __m256i b, __m256i c, uint32_t mode, bool saturate) {
  using F = goc::FixupFormat<Width>;
  using O = Ops<Width == 64>;
  auto zero = O::set(0), inf = O::set(F::infinity), signbit = O::set(F::sign),
       magmask = O::set(F::sign - 1);
  auto pm = _mm256_and_si256(a, magmask), bm = _mm256_and_si256(b, magmask),
       cm = _mm256_and_si256(c, magmask);
  auto sign = _mm256_and_si256(_mm256_xor_si256(b, c), signbit);
  auto overflow = O::set(F::infinity - (Width == 16 && saturate));
  auto result = _mm256_or_si256(sign, O::select(O::gt(inf, pm), pm, overflow));
  if constexpr (Width != 16) {
    auto delta = O::sub(O::shr(cm, F::fraction), O::shr(bm, F::fraction));
    result =
        O::select(O::gt(O::set(uint64_t(-int64_t(F::bias + F::fraction))), delta), sign, result);
  }
  auto bz = O::eq(bm, zero), cz = O::eq(cm, zero), bi = O::eq(bm, inf), ci = O::eq(cm, inf);
  result = O::select(O::either(cz, bi), sign, result);
  result = O::select(O::either(bz, ci), _mm256_or_si256(sign, inf), result);
  result = O::select(O::either(O::both(bz, cz), O::both(bi, ci)),
                     O::set(F::sign | F::infinity | F::quiet), result);
  result = O::select(O::gt(bm, inf), _mm256_or_si256(b, O::set(F::quiet)), result);
  result = O::select(O::gt(cm, inf), _mm256_or_si256(c, O::set(F::quiet)), result);
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
  using F = goc::FixupFormat<Width>;
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
    if constexpr (Width == 16) {
      unsigned shift = mode & (GOC_ALU_HIGH_A << operand) ? 16 : 0;
      result =
          _mm256_and_si256(_mm256_srl_epi32(result, _mm_cvtsi32_si128(int(shift))), O::set(65535));
    }
  }
  auto keep = O::set(mode & (GOC_ALU_ABS_A << operand) ? F::sign - 1 : ~uint64_t(0));
  auto flip = O::set(mode & (GOC_ALU_NEG_A << operand) ? F::sign : 0);
  return _mm256_xor_si256(_mm256_and_si256(result, keep), flip);
}

} // namespace

namespace goc {

template <unsigned Width>
void fixup_x86_64_v3(uint32_t mask, uint32_t mode, bool saturate, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  constexpr unsigned lanes = Width == 64 ? 4 : 8;
  uint32_t staged[2][32];
  for (unsigned lane = 0; lane < 32; lane += lanes) {
    auto result = value<Width>(load<Width>(a, lane, mode, 0), load<Width>(b, lane, mode, 1),
                               load<Width>(c, lane, mode, 2), mode, saturate);
    if constexpr (Width == 64) {
      auto words = _mm256_permutevar8x32_epi32(result, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
      _mm_storeu_si128(reinterpret_cast<__m128i *>(staged[0] + lane),
                       _mm256_castsi256_si128(words));
      _mm_storeu_si128(reinterpret_cast<__m128i *>(staged[1] + lane),
                       _mm256_extracti128_si256(words, 1));
    } else {
      if constexpr (Width == 16) {
        unsigned shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
        auto old = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d[0] + lane));
        result = _mm256_or_si256(_mm256_andnot_si256(_mm256_set1_epi32(int(65535u << shift)), old),
                                 _mm256_sll_epi32(result, _mm_cvtsi32_si128(int(shift))));
      }
      auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(mask >> lane)),
                                      _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
      _mm256_maskstore_epi32(reinterpret_cast<int *>(d[0] + lane), active, result);
    }
  }
  // Delay FP64 writes for cross-half aliases; D1 wins when D0 and D1 alias.
  if constexpr (Width == 64)
    for (unsigned reg = 0; reg < 2; ++reg)
      for (unsigned lane = 0; lane < 32; lane += 8) {
        auto result = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(staged[reg] + lane));
        auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(mask >> lane)),
                                        _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
        _mm256_maskstore_epi32(reinterpret_cast<int *>(d[reg] + lane), active, result);
      }
}

template void fixup_x86_64_v3<16>(uint32_t, uint32_t, bool, uint32_t *const *,
                                  const uint32_t *const *, const uint32_t *const *,
                                  const uint32_t *const *);
template void fixup_x86_64_v3<32>(uint32_t, uint32_t, bool, uint32_t *const *,
                                  const uint32_t *const *, const uint32_t *const *,
                                  const uint32_t *const *);
template void fixup_x86_64_v3<64>(uint32_t, uint32_t, bool, uint32_t *const *,
                                  const uint32_t *const *, const uint32_t *const *,
                                  const uint32_t *const *);

} // namespace goc

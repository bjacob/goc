// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Vectorized form of the rocjitsu-derived division pre-scaling model.

#include "goc/goc.h"
#include "rdna4_div_scale.h"
#include "rdna4_division.h"
#include "x86_64/rdna4_division_simd.h"
#include "x86_64/rdna4_division_x86_64_v3.h"

#include <immintrin.h>
#include <stdint.h>

namespace {

template <bool Wide> using Ops = goc::DivisionOpsV3<Wide>;

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
    auto result = division_scale_value<Width, Ops<Width == 64>>(
        load<Width>(a, lane, mode, 0), load<Width>(b, lane, mode, 1), load<Width>(c, lane, mode, 2),
        mode, post);
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

// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Vectorized form of the rocjitsu-derived division pre-scaling model.

#include "goc/goc.h"
#include "rdna4_div_scale.h"
#include "rdna4_division.h"
#include "x86_64/rdna4_division_simd.h"
#include "x86_64/rdna4_division_x86_64_v4.h"

#include <immintrin.h>
#include <stdint.h>

namespace {

template <bool Wide> using Ops = goc::DivisionOpsV4<Wide>;

template <unsigned Width>
__m512i load(const uint32_t *const *v, unsigned lane, uint32_t mode, unsigned operand) {
  using F = goc::DivisionFormat<Width>;
  using O = Ops<Width == 64>;
  __m512i result;
  if constexpr (Width == 64) {
    auto low =
        _mm512_cvtepu32_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(v[0] + lane)));
    auto high =
        _mm512_cvtepu32_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(v[1] + lane)));
    result = _mm512_or_si512(low, _mm512_slli_epi64(high, 32));
  } else {
    result = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(v[0] + lane));
  }
  auto flip = O::set(mode & (GOC_ALU_NEG_A << operand) ? F::sign : 0);
  return _mm512_xor_si512(result, flip);
}

} // namespace

namespace goc {

template <unsigned Width>
uint32_t div_scale_x86_64_v4(uint32_t mask, uint32_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c) {
  constexpr unsigned lanes = Width == 64 ? 8 : 16;
  uint32_t staged[2][32], conditions = 0;
  for (unsigned lane = 0; lane < 32; lane += lanes) {
    typename Ops<Width == 64>::Mask post;
    auto result = division_scale_value<Width, Ops<Width == 64>>(
        load<Width>(a, lane, mode, 0), load<Width>(b, lane, mode, 1), load<Width>(c, lane, mode, 2),
        mode, post);
    conditions |= uint32_t(post) << lane;
    if constexpr (Width == 64) {
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(staged[0] + lane),
                          _mm512_cvtepi64_epi32(result));
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(staged[1] + lane),
                          _mm512_cvtepi64_epi32(_mm512_srli_epi64(result, 32)));
    } else {
      _mm512_mask_storeu_epi32(d[0] + lane, __mmask16(mask >> lane), result);
    }
  }
  // Preserve cross-half aliases by committing D0 then D1 after all reads.
  if constexpr (Width == 64)
    for (unsigned reg = 0; reg < 2; ++reg)
      for (unsigned lane = 0; lane < 32; lane += 16) {
        auto result = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(staged[reg] + lane));
        _mm512_mask_storeu_epi32(d[reg] + lane, __mmask16(mask >> lane), result);
      }
  return conditions & mask;
}

template uint32_t div_scale_x86_64_v4<32>(uint32_t, uint32_t, uint32_t *const *,
                                          const uint32_t *const *, const uint32_t *const *,
                                          const uint32_t *const *);
template uint32_t div_scale_x86_64_v4<64>(uint32_t, uint32_t, uint32_t *const *,
                                          const uint32_t *const *, const uint32_t *const *,
                                          const uint32_t *const *);

} // namespace goc

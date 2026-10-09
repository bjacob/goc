// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Vectorized form of the rocjitsu-derived division fixup model.

#include "goc/goc.h"
#include "rdna4_div_fixup.h"
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
    if constexpr (Width == 16) {
      unsigned shift = mode & (GOC_ALU_HIGH_A << operand) ? 16 : 0;
      result =
          _mm512_and_si512(_mm512_srl_epi32(result, _mm_cvtsi32_si128(int(shift))), O::set(65535));
    }
  }
  auto keep = O::set(mode & (GOC_ALU_ABS_A << operand) ? F::sign - 1 : ~uint64_t(0));
  auto flip = O::set(mode & (GOC_ALU_NEG_A << operand) ? F::sign : 0);
  return _mm512_xor_si512(_mm512_and_si512(result, keep), flip);
}

} // namespace

namespace goc {

template <unsigned Width>
void fixup_x86_64_v4(uint32_t mask, uint32_t mode, bool saturate, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  constexpr unsigned lanes = Width == 64 ? 8 : 16;
  uint32_t staged[2][32];
  for (unsigned lane = 0; lane < 32; lane += lanes) {
    auto result = division_fixup_value<Width, Ops<Width == 64>>(
        load<Width>(a, lane, mode, 0), load<Width>(b, lane, mode, 1), load<Width>(c, lane, mode, 2),
        mode, saturate);
    if constexpr (Width == 64) {
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(staged[0] + lane),
                          _mm512_cvtepi64_epi32(result));
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(staged[1] + lane),
                          _mm512_cvtepi64_epi32(_mm512_srli_epi64(result, 32)));
    } else {
      if constexpr (Width == 16) {
        unsigned shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
        auto old = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(d[0] + lane));
        result = _mm512_or_si512(_mm512_andnot_si512(_mm512_set1_epi32(int(65535u << shift)), old),
                                 _mm512_sll_epi32(result, _mm_cvtsi32_si128(int(shift))));
      }
      _mm512_mask_storeu_epi32(d[0] + lane, __mmask16(mask >> lane), result);
    }
  }
  // Delay FP64 writes for cross-half aliases; D1 wins when D0 and D1 alias.
  if constexpr (Width == 64)
    for (unsigned reg = 0; reg < 2; ++reg)
      for (unsigned lane = 0; lane < 32; lane += 16) {
        auto result = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(staged[reg] + lane));
        _mm512_mask_storeu_epi32(d[reg] + lane, __mmask16(mask >> lane), result);
      }
}

template void fixup_x86_64_v4<16>(uint32_t, uint32_t, bool, uint32_t *const *,
                                  const uint32_t *const *, const uint32_t *const *,
                                  const uint32_t *const *);
template void fixup_x86_64_v4<32>(uint32_t, uint32_t, bool, uint32_t *const *,
                                  const uint32_t *const *, const uint32_t *const *,
                                  const uint32_t *const *);
template void fixup_x86_64_v4<64>(uint32_t, uint32_t, bool, uint32_t *const *,
                                  const uint32_t *const *, const uint32_t *const *,
                                  const uint32_t *const *);

} // namespace goc

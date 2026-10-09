// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_mixed_fma.h"
#include "rdna4_mixed_fma_scalar.h"
#include "x86_64/rdna4_half_x86_64_v3.h"
#include "x86_64/rdna4_mixed_half_x86_64_v3.h"

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

} // namespace

namespace goc {

void mixed_fma_float_x86_64_v3(uint32_t mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                               const uint32_t *b, const uint32_t *c) {
  for (int lane = 0; lane < 32; lane += 8) {
    auto x = input8(a + lane, mode), y = input8(b + lane, mode >> 1),
         z = input8(c + lane, mode >> 2);
    auto result = _mm256_fmadd_ps(x, y, z);
    if (mode & GOC_ALU_CLAMP)
      result = _mm256_min_ps(_mm256_max_ps(result, _mm256_setzero_ps()), _mm256_set1_ps(1));
    auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(mask >> lane)),
                                    _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), active, _mm256_castps_si256(result));
  }
}

void mixed_fma_half_x86_64_v3(bool high, bool saturate, uint32_t mask, uint32_t mode, uint32_t *d,
                              const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  for (int lane = 0; lane < 32; lane += 4) {
    auto x = input4(a + lane, mode), y = input4(b + lane, mode >> 1),
         z = input4(c + lane, mode >> 2);
    auto result = mixed_half_value4<false>(x, y, z, saturate, mode & GOC_ALU_CLAMP);
    unsigned shift = high ? 16 : 0;
    auto original = _mm_loadu_si128(reinterpret_cast<const __m128i *>(d + lane));
    auto output_mask = _mm_set1_epi32(int(uint32_t(65535) << shift));
    result = _mm_or_si128(_mm_sll_epi32(result, _mm_cvtsi32_si128(shift)),
                          _mm_andnot_si128(output_mask, original));
    auto active = _mm_sllv_epi32(_mm_set1_epi32(int(mask >> lane)), _mm_setr_epi32(31, 30, 29, 28));
    _mm_maskstore_epi32(reinterpret_cast<int *>(d + lane), active, result);
  }
}

} // namespace goc

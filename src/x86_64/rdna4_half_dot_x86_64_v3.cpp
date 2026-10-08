// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_simd.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {
namespace {

template <bool Bf16> __m256 input(__m256i words, int shift, uint32_t mode) {
  auto halves = _mm256_and_si256(_mm256_srl_epi32(words, _mm_cvtsi32_si128(shift)),
                                 _mm256_set1_epi32(mode & GOC_ALU_ABS_A ? 0x7fff : 0xffff));
  halves = _mm256_xor_si256(halves, _mm256_set1_epi32(mode & GOC_ALU_NEG_A ? 0x8000 : 0));
  if constexpr (Bf16) {
    auto zero_exp = _mm256_cmpeq_epi32(_mm256_and_si256(halves, _mm256_set1_epi32(0x7f80)),
                                       _mm256_setzero_si256());
    halves =
        _mm256_blendv_epi8(halves, _mm256_and_si256(halves, _mm256_set1_epi32(0x8000)), zero_exp);
    return _mm256_castsi256_ps(_mm256_slli_epi32(halves, 16));
  } else
    return _mm256_cvtph_ps(
        _mm_packus_epi32(_mm256_castsi256_si128(halves), _mm256_extracti128_si256(halves, 1)));
}

template <bool Bf16> __m256i narrow(__m256 value, bool saturate) {
  auto bits = _mm256_castps_si256(value);
  auto magnitude = _mm256_and_si256(bits, _mm256_set1_epi32(INT32_MAX));
  if constexpr (Bf16) {
    auto tie = _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1));
    auto result = _mm256_srli_epi32(
        _mm256_add_epi32(bits, _mm256_add_epi32(tie, _mm256_set1_epi32(0x7fff))), 16);
    auto nan = _mm256_cmpgt_epi32(magnitude, _mm256_set1_epi32(0x7f800000));
    result = _mm256_blendv_epi8(
        result, _mm256_or_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(0x40)), nan);
    auto zero_exp = _mm256_cmpeq_epi32(_mm256_and_si256(result, _mm256_set1_epi32(0x7f80)),
                                       _mm256_setzero_si256());
    return _mm256_blendv_epi8(result, _mm256_and_si256(result, _mm256_set1_epi32(0x8000)),
                              zero_exp);
  } else {
    auto result = _mm256_cvtepu16_epi32(
        _mm256_cvtps_ph(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    if (saturate) {
      auto finite = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x7f800000), magnitude);
      auto infinity = _mm256_cmpeq_epi32(_mm256_and_si256(result, _mm256_set1_epi32(0x7fff)),
                                         _mm256_set1_epi32(0x7c00));
      result = _mm256_add_epi32(result, _mm256_and_si256(finite, infinity));
    }
    return result;
  }
}

template <bool Bf16>
void dot(bool saturate, uint32_t mask, uint32_t mode, uint32_t *d, const uint32_t *a,
         const uint32_t *b, const uint32_t *c) {
  int c_shift = mode & GOC_ALU_HIGH_C ? 16 : 0;
  int d_shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  auto keep = _mm256_set1_epi32(d_shift ? 65535 : -65536);
  for (int lane = 0; lane < 32; lane += 8) {
    auto va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    auto vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
    auto vc = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c + lane));
    auto p0 = _mm256_mul_ps(input<Bf16>(va, 0, mode), input<Bf16>(vb, 0, mode >> 1));
    auto p1 = _mm256_mul_ps(input<Bf16>(va, 16, mode), input<Bf16>(vb, 16, mode >> 1));
    auto value = _mm256_add_ps(_mm256_add_ps(p0, p1), input<Bf16>(vc, c_shift, mode >> 2));
    auto result = _mm256_sll_epi32(narrow<Bf16>(value, saturate), _mm_cvtsi32_si128(d_shift));
    auto old = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d + lane));
    result = _mm256_or_si256(result, _mm256_and_si256(old, keep));
    auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(mask >> lane)),
                                    _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), active, result);
  }
}

} // namespace

void half_dot_x86_64_v3(bool bf16, bool saturate, uint32_t mask, uint32_t mode, uint32_t *d,
                        const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  if (bf16)
    dot<true>(saturate, mask, mode, d, a, b, c);
  else
    dot<false>(saturate, mask, mode, d, a, b, c);
}

} // namespace goc

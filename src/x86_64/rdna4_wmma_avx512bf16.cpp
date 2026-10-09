// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_simd.h"

#include <cstring>
#include <immintrin.h>
#include <stdint.h>

namespace {

// Reduce pairs of unsigned 16-bit values to unsigned 32-bit lanes, then reduce
// across lanes. Only the low halfword of each intermediate lane is retained.
int min_exponent(__m512i value) {
  value = _mm512_min_epu16(value, _mm512_srli_epi32(value, 16));
  return int(_mm512_reduce_min_epu32(_mm512_and_si512(value, _mm512_set1_epi32(65535))));
}

int max_exponent(__m512i value) {
  value = _mm512_max_epu16(value, _mm512_srli_epi32(value, 16));
  return int(_mm512_reduce_max_epu32(_mm512_and_si512(value, _mm512_set1_epi32(65535))));
}

} // namespace

namespace goc {

bool wmma_inputs_avx512bf16(const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  int minimum[2], maximum[2];
  for (int operand = 0; operand < 2; ++operand) {
    const uint32_t *const *v = operand ? b : a;
    __m512i min_exp = _mm512_set1_epi16(255), max_exp = _mm512_setzero_si512();
    for (int reg = 0; reg < 4; ++reg)
      for (int lane = 0; lane < 32; lane += 16) {
        __m512i magnitude =
            _mm512_and_si512(_mm512_loadu_si512(v[reg] + lane), _mm512_set1_epi16(0x7fff));
        __mmask32 zero = _mm512_cmpeq_epi16_mask(magnitude, _mm512_setzero_si512());
        __mmask32 tiny = _mm512_cmp_epu16_mask(magnitude, _mm512_set1_epi16(0x80), _MM_CMPINT_LT);
        __mmask32 special =
            _mm512_cmp_epu16_mask(magnitude, _mm512_set1_epi16(0x7f80), _MM_CMPINT_GE);
        if (special || (tiny & ~zero))
          return false;
        __m512i exponent = _mm512_srli_epi16(magnitude, 7);
        max_exp = _mm512_max_epu16(max_exp, exponent);
        // Zero factors do not lower the minimum nonzero exponent.
        min_exp = _mm512_min_epu16(min_exp,
                                   _mm512_mask_mov_epi16(exponent, zero, _mm512_set1_epi16(255)));
      }
    minimum[operand] = min_exponent(min_exp);
    maximum[operand] = max_exponent(max_exp);
  }

  // Preserve the previous conservative bounds on every possible product.
  if (minimum[0] + minimum[1] < 128 || maximum[0] + maximum[1] > 380)
    return false;
  for (int reg = 0; reg < 8; ++reg)
    for (int lane = 0; lane < 32; lane += 16) {
      __m512i magnitude =
          _mm512_and_si512(_mm512_loadu_si512(c[reg] + lane), _mm512_set1_epi32(0x7fffffff));
      __mmask16 zero = _mm512_cmpeq_epi32_mask(magnitude, _mm512_setzero_si512());
      __mmask16 tiny =
          _mm512_cmp_epu32_mask(magnitude, _mm512_set1_epi32(0x00800000), _MM_CMPINT_LT);
      __mmask16 special =
          _mm512_cmp_epu32_mask(magnitude, _mm512_set1_epi32(0x7f800000), _MM_CMPINT_GE);
      if (special || (tiny & ~zero))
        return false;
    }
  return true;
}

template <bool Modified>
static void wmma_bf16(uint32_t modifiers, uint32_t *const *d, const uint32_t *const *a,
                      const uint32_t *const *b, const uint32_t *const *c) {
  const uint32_t a_sign = ((modifiers & GOC_WMMA_NEG_LO_A) ? 0x8000U : 0) |
                          ((modifiers & GOC_WMMA_NEG_HI_A) ? 0x80000000U : 0);
  const __m512i b_sign = _mm512_set1_epi32(((modifiers & GOC_WMMA_NEG_LO_B) ? 0x8000 : 0) |
                                           ((modifiers & GOC_WMMA_NEG_HI_B) ? INT32_MIN : 0));
  uint32_t result[8][32];
  for (int row = 0; row < 16; ++row) {
    int reg = row % 8, group = row / 8;
    __m512 acc = _mm512_castsi512_ps(_mm512_loadu_si512(c[reg] + 16 * group));
    if constexpr (Modified) {
      __m512i keep = _mm512_set1_epi32((modifiers & GOC_WMMA_ABS_C) ? INT32_MAX : -1);
      __m512i sign = _mm512_set1_epi32((modifiers & GOC_WMMA_NEG_C) ? INT32_MIN : 0);
      acc = _mm512_castsi512_ps(
          _mm512_xor_si512(_mm512_and_si512(_mm512_castps_si512(acc), keep), sign));
    }
    for (int pair = 0; pair < 8; ++pair) {
      uint32_t aw = a[pair % 4][row + 16 * (pair / 4)];
      if constexpr (Modified)
        aw ^= a_sign;
      int32_t signed_word;
      std::memcpy(&signed_word, &aw, sizeof(aw));
      __m512bh va = (__m512bh)_mm512_set1_epi32(signed_word);
      __m512i bw = _mm512_loadu_si512(b[pair % 4] + 16 * (pair / 4));
      if constexpr (Modified)
        bw = _mm512_xor_si512(bw, b_sign);
      __m512bh vb = (__m512bh)bw;
      acc = _mm512_dpbf16_ps(acc, va, vb);
    }
    _mm512_storeu_si512(result[reg] + 16 * group, _mm512_castps_si512(acc));
  }
  for (int reg = 0; reg < 8; ++reg)
    for (int group = 0; group < 2; ++group)
      _mm512_storeu_si512(d[reg] + 16 * group, _mm512_loadu_si512(result[reg] + 16 * group));
}

void wmma_avx512bf16(uint32_t modifiers, uint32_t *const *d, const uint32_t *const *a,
                     const uint32_t *const *b, const uint32_t *const *c) {
  if (modifiers)
    wmma_bf16<true>(modifiers, d, a, b, c);
  else
    wmma_bf16<false>(0, d, a, b, c);
}

} // namespace goc

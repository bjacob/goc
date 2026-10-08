// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_fma.h"
#include "rdna4_simd.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {

namespace {

template <bool Dx9Zero, FmaOperands Operands = FmaOperands::Registers>
void run(uint32_t mask, uint32_t modifiers, uint32_t *d, const uint32_t *a, const uint32_t *b,
         const uint32_t *c, uint32_t literal = 0) {
  const __m256i keep_a = _mm256_set1_epi32((modifiers & GOC_ALU_ABS_A) ? 0x7fffffff : -1);
  const __m256i flip_a = _mm256_set1_epi32((modifiers & GOC_ALU_NEG_A) ? INT32_MIN : 0);
  const __m256i keep_b = _mm256_set1_epi32((modifiers & GOC_ALU_ABS_B) ? 0x7fffffff : -1);
  const __m256i flip_b = _mm256_set1_epi32((modifiers & GOC_ALU_NEG_B) ? INT32_MIN : 0);
  const __m256i keep_c = _mm256_set1_epi32((modifiers & GOC_ALU_ABS_C) ? 0x7fffffff : -1);
  const __m256i flip_c = _mm256_set1_epi32((modifiers & GOC_ALU_NEG_C) ? INT32_MIN : 0);
  const float scales[] = {1, 2, 4, 0.5f};
  const auto scale = _mm256_set1_ps(scales[(modifiers >> 6) & 3]);
  for (int i = 0; i < 32; i += 8) {
    auto va = _mm256_castsi256_ps(_mm256_xor_si256(
        _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i)), keep_a),
        flip_a));
    auto vb = _mm256_castsi256_ps(_mm256_xor_si256(
        _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i)), keep_b),
        flip_b));
    __m256 vc;
    if constexpr (Operands == FmaOperands::MultiplyLiteral) {
      vc = vb;
      vb = _mm256_castsi256_ps(_mm256_set1_epi32(int(literal)));
    } else if constexpr (Operands == FmaOperands::AddLiteral) {
      vc = _mm256_castsi256_ps(_mm256_set1_epi32(int(literal)));
    } else {
      vc = _mm256_castsi256_ps(_mm256_xor_si256(
          _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(c + i)), keep_c),
          flip_c));
    }
    auto result = _mm256_fmadd_ps(va, vb, vc);
    if constexpr (Dx9Zero) {
      auto zero = _mm256_setzero_ps();
      auto has_zero =
          _mm256_or_ps(_mm256_cmp_ps(va, zero, _CMP_EQ_OQ), _mm256_cmp_ps(vb, zero, _CMP_EQ_OQ));
      result = _mm256_blendv_ps(result, vc, has_zero);
    }
    if (modifiers & GOC_ALU_OMOD_HALF)
      result = _mm256_mul_ps(result, scale);
    if (modifiers & GOC_ALU_CLAMP)
      result = _mm256_min_ps(_mm256_max_ps(result, _mm256_setzero_ps()), _mm256_set1_ps(1));
    __m256i active = _mm256_sllv_epi32(_mm256_set1_epi32(int(mask >> i)),
                                       _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + i), active, _mm256_castps_si256(result));
  }
}

} // namespace

void fma_x86_64_v3(uint32_t mask, uint32_t modifiers, uint32_t *d, const uint32_t *a,
                   const uint32_t *b, const uint32_t *c) {
  run<false>(mask, modifiers, d, a, b, c);
}

void fma_dx9_zero_x86_64_v3(uint32_t mask, uint32_t modifiers, uint32_t *d, const uint32_t *a,
                            const uint32_t *b, const uint32_t *c) {
  run<true>(mask, modifiers, d, a, b, c);
}

template <bool Multiply>
void literal_fma_x86_64_v3(uint32_t mask, uint32_t literal, uint32_t *d, const uint32_t *a,
                           const uint32_t *b) {
  run<false, Multiply ? FmaOperands::MultiplyLiteral : FmaOperands::AddLiteral>(mask, 0, d, a, b,
                                                                                nullptr, literal);
}

template void literal_fma_x86_64_v3<false>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                           const uint32_t *);
template void literal_fma_x86_64_v3<true>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                          const uint32_t *);

} // namespace goc

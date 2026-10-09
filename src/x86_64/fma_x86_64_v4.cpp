// SPDX-License-Identifier: MIT

#include "fma.h"
#include "goc/goc.h"
#include "simd.h"
#include "x86_64/alu_x86_64_v4.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {

namespace {

template <bool Dx9Zero, FmaOperands Operands = FmaOperands::Registers>
void run(uint32_t exec_mask, uint32_t modifiers, uint32_t *d, const uint32_t *a, const uint32_t *b,
         const uint32_t *c, uint32_t literal = 0) {
  const __m512i keep_a = _mm512_set1_epi32((modifiers & GOC_ALU_ABS_A) ? 0x7fffffff : -1);
  const __m512i flip_a = _mm512_set1_epi32((modifiers & GOC_ALU_NEG_A) ? INT32_MIN : 0);
  const __m512i keep_b = _mm512_set1_epi32((modifiers & GOC_ALU_ABS_B) ? 0x7fffffff : -1);
  const __m512i flip_b = _mm512_set1_epi32((modifiers & GOC_ALU_NEG_B) ? INT32_MIN : 0);
  const __m512i keep_c = _mm512_set1_epi32((modifiers & GOC_ALU_ABS_C) ? 0x7fffffff : -1);
  const __m512i flip_c = _mm512_set1_epi32((modifiers & GOC_ALU_NEG_C) ? INT32_MIN : 0);
  const float scales[] = {1, 2, 4, 0.5f};
  const auto scale = _mm512_set1_ps(scales[(modifiers >> 6) & 3]);
  for (int i = 0; i < 32; i += 16) {
    auto va = _mm512_castsi512_ps(
        _mm512_xor_si512(_mm512_and_si512(_mm512_loadu_si512(a + i), keep_a), flip_a));
    auto vb = _mm512_castsi512_ps(
        _mm512_xor_si512(_mm512_and_si512(_mm512_loadu_si512(b + i), keep_b), flip_b));
    __m512 vc;
    if constexpr (Operands == FmaOperands::MultiplyLiteral) {
      vc = vb;
      vb = _mm512_castsi512_ps(_mm512_set1_epi32(int(literal)));
    } else if constexpr (Operands == FmaOperands::AddLiteral) {
      vc = _mm512_castsi512_ps(_mm512_set1_epi32(int(literal)));
    } else {
      vc = _mm512_castsi512_ps(
          _mm512_xor_si512(_mm512_and_si512(_mm512_loadu_si512(c + i), keep_c), flip_c));
    }
    if constexpr (Dx9Zero) {
      va = flush_denorm_f32(va);
      vb = flush_denorm_f32(vb);
      vc = flush_denorm_f32(vc);
      auto zero = _mm512_setzero_ps();
      auto has_zero =
          _mm512_cmp_ps_mask(va, zero, _CMP_EQ_OQ) | _mm512_cmp_ps_mask(vb, zero, _CMP_EQ_OQ);
      va = _mm512_mask_mov_ps(va, has_zero, zero);
      vb = _mm512_mask_mov_ps(vb, has_zero, _mm512_set1_ps(1));
    }
    auto result = _mm512_fmadd_ps(va, vb, vc);
    if constexpr (Dx9Zero)
      result = flush_denorm_f32(result);
    if (modifiers & GOC_ALU_OMOD_HALF) {
      result = prepare_omod_f32(result, modifiers);
      result = _mm512_mul_ps(result, scale);
    }
    if (modifiers & GOC_ALU_CLAMP)
      result = _mm512_min_ps(_mm512_max_ps(result, _mm512_setzero_ps()), _mm512_set1_ps(1));
    _mm512_mask_storeu_epi32(d + i, static_cast<__mmask16>(exec_mask >> i),
                             _mm512_castps_si512(result));
  }
}

} // namespace

void fma_x86_64_v4(uint32_t exec_mask, uint32_t modifiers, uint32_t *d, const uint32_t *a,
                   const uint32_t *b, const uint32_t *c) {
  run<false>(exec_mask, modifiers, d, a, b, c);
}

void fma_dx9_zero_x86_64_v4(uint32_t exec_mask, uint32_t modifiers, uint32_t *d, const uint32_t *a,
                            const uint32_t *b, const uint32_t *c) {
  run<true>(exec_mask, modifiers, d, a, b, c);
}

template <bool Multiply>
void literal_fma_x86_64_v4(uint32_t exec_mask, uint32_t literal, uint32_t *d, const uint32_t *a,
                           const uint32_t *b) {
  run<false, Multiply ? FmaOperands::MultiplyLiteral : FmaOperands::AddLiteral>(
      exec_mask, 0, d, a, b, nullptr, literal);
}

template void literal_fma_x86_64_v4<false>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                           const uint32_t *);
template void literal_fma_x86_64_v4<true>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                          const uint32_t *);

} // namespace goc

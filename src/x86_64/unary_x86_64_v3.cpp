// SPDX-License-Identifier: MIT

#include "x86_64/unary_x86_64_v3.h"
#include "goc/goc.h"
#include "unary.h"
#include "x86_64/alu_x86_64_v3.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {
namespace {

template <Unary Op>
void run(uint32_t exec_mask, uint32_t modifiers, uint32_t *d, const uint32_t *a) {
  const __m256i keep = _mm256_set1_epi32((modifiers & GOC_ALU_ABS_A) ? 0x7fffffff : -1);
  const __m256i flip = _mm256_set1_epi32((modifiers & GOC_ALU_NEG_A) ? INT32_MIN : 0);
  const float scales[] = {1, 2, 4, 0.5f};
  const __m256 scale = _mm256_set1_ps(scales[(modifiers >> 6) & 3]);
  for (int lane = 0; lane < 32; lane += 8) {
    __m256 value = _mm256_castsi256_ps(_mm256_xor_si256(
        _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane)), keep),
        flip));
    if constexpr (unary_flushes_f32<Op>)
      value = flush_denorm_f32(value);
    value = unary_value<Op>(value);
    if constexpr (unary_flushes_f32<Op>)
      value = flush_denorm_f32(value);
    if (modifiers & GOC_ALU_OMOD_HALF) {
      value = prepare_omod_f32(value, modifiers);
      value = _mm256_mul_ps(value, scale);
    }
    if (modifiers & GOC_ALU_CLAMP)
      value = _mm256_min_ps(_mm256_max_ps(value, _mm256_setzero_ps()), _mm256_set1_ps(1));
    __m256i lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                               _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask,
                           _mm256_castps_si256(value));
  }
}

} // namespace

void unary_x86_64_v3(Unary op, uint32_t exec_mask, uint32_t modifiers, uint32_t *d,
                     const uint32_t *a) {
  switch (op) {
  case Unary::Fract:
    return run<Unary::Fract>(exec_mask, modifiers, d, a);
  case Unary::FrexpMant:
    return run<Unary::FrexpMant>(exec_mask, modifiers, d, a);
  case Unary::Trunc:
    return run<Unary::Trunc>(exec_mask, modifiers, d, a);
  case Unary::Ceil:
    return run<Unary::Ceil>(exec_mask, modifiers, d, a);
  case Unary::Rndne:
    return run<Unary::Rndne>(exec_mask, modifiers, d, a);
  case Unary::Floor:
    return run<Unary::Floor>(exec_mask, modifiers, d, a);
  case Unary::Sqrt:
    return run<Unary::Sqrt>(exec_mask, modifiers, d, a);
  case Unary::Rcp:
    return run<Unary::Rcp>(exec_mask, modifiers, d, a);
  case Unary::Rsq:
    return run<Unary::Rsq>(exec_mask, modifiers, d, a);
  case Unary::Exp:
  case Unary::Log:
    return;
  }
}

} // namespace goc

// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "half_unary.h"
#include "unary.h"
#include "x86_64/half_x86_64_v3.h"
#include "x86_64/unary_x86_64_v3.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {

template <Unary Op>
void half_unary_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                          const uint32_t *a) {
  bool saturate = flags & GOC_FP16_OVFL;
  int a_shift = mode & GOC_ALU_HIGH_A ? 16 : 0;
  int d_shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  auto keep = _mm256_set1_epi32(d_shift ? 65535 : -65536);
  const float scales[] = {1, 2, 4, 0.5f};
  auto scale = _mm256_set1_ps(scales[(mode >> 6) & 3]);
  for (int lane = 0; lane < 32; lane += 8) {
    auto x = half_input<false>(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane)),
                               a_shift, mode);
    if constexpr (unary_rounds_f32<Op> || Op == Unary::FrexpMant || Op == Unary::Fract ||
                  half_unary_sfu<Op>)
      if (flags & GOC_FP_FLUSH_INPUT_DENORMALS) {
        auto bits = _mm256_castps_si256(x);
        auto magnitude = _mm256_and_si256(bits, _mm256_set1_epi32(0x7fffffff));
        auto tiny = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x38800000), magnitude);
        auto sign = _mm256_and_si256(bits, _mm256_set1_epi32(int(0x80000000U)));
        x = _mm256_castsi256_ps(_mm256_blendv_epi8(bits, sign, tiny));
      }
    auto value = unary_value<Op, true>(x);
    if constexpr (Op == Unary::Fract || half_unary_sfu<Op>)
      if (flags & GOC_FP_FLUSH_OUTPUT_DENORMALS) {
        auto sign = _mm256_and_ps(value, _mm256_set1_ps(-0.0f));
        auto magnitude = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), value);
        auto tiny = _mm256_cmp_ps(magnitude, _mm256_set1_ps(0x1p-14f), _CMP_LT_OQ);
        value = _mm256_blendv_ps(value, sign, tiny);
      }
    if constexpr (Op == Unary::Log) {
      if (saturate)
        value = _mm256_blendv_ps(value, _mm256_set1_ps(-65504),
                                 _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_EQ_OQ));
    }
    if constexpr (Op == Unary::Rcp || Op == Unary::Rsq)
      if (saturate) {
        auto maximum =
            _mm256_or_ps(_mm256_set1_ps(65504.0f), _mm256_and_ps(value, _mm256_set1_ps(-0.0f)));
        value = _mm256_blendv_ps(value, maximum, _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_EQ_OQ));
      }
    if constexpr (Op == Unary::Exp || Op == Unary::Log || half_unary_sfu<Op>) {
      // CLAMP makes intermediate SFU overflow irrelevant; normal half rounding
      // commutes with the power-of-two output scale.
      if ((mode & GOC_ALU_OMOD_HALF) && (!half_unary_sfu<Op> || !(mode & GOC_ALU_CLAMP)))
        value = half_input<false>(half_narrow<false>(value, saturate), 0, 0);
    }
    if (mode & GOC_ALU_OMOD_HALF)
      value = _mm256_mul_ps(prepare_omod_f16(value, mode), scale);
    if (mode & GOC_ALU_CLAMP)
      value = _mm256_min_ps(_mm256_max_ps(value, _mm256_setzero_ps()), _mm256_set1_ps(1));
    auto result = _mm256_sll_epi32(half_narrow<false>(value, saturate), _mm_cvtsi32_si128(d_shift));
    auto old = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d + lane));
    result = _mm256_or_si256(result, _mm256_and_si256(old, keep));
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                            _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask, result);
  }
}

template void half_unary_x86_64_v3<Unary::Trunc>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                 const uint32_t *);
template void half_unary_x86_64_v3<Unary::Ceil>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                const uint32_t *);
template void half_unary_x86_64_v3<Unary::Rndne>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                 const uint32_t *);
template void half_unary_x86_64_v3<Unary::Floor>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                 const uint32_t *);
template void half_unary_x86_64_v3<Unary::Sqrt>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                const uint32_t *);
template void half_unary_x86_64_v3<Unary::Rcp>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                               const uint32_t *);
template void half_unary_x86_64_v3<Unary::Rsq>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                               const uint32_t *);
template void half_unary_x86_64_v3<Unary::Fract>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                 const uint32_t *);
template void half_unary_x86_64_v3<Unary::FrexpMant>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                     const uint32_t *);

template void half_unary_x86_64_v3<Unary::Exp>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                               const uint32_t *);
template void half_unary_x86_64_v3<Unary::Log>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                               const uint32_t *);

} // namespace goc

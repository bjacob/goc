// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_half_unary.h"
#include "rdna4_unary.h"
#include "x86_64/rdna4_half_x86_64_v3.h"
#include "x86_64/rdna4_unary_x86_64_v3.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {

template <Unary Op>
void half_unary_x86_64_v3(bool saturate, uint32_t mask, uint32_t mode, uint32_t *d,
                          const uint32_t *a) {
  int a_shift = mode & GOC_ALU_HIGH_A ? 16 : 0;
  int d_shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  auto keep = _mm256_set1_epi32(d_shift ? 65535 : -65536);
  const float scales[] = {1, 2, 4, 0.5f};
  auto scale = _mm256_set1_ps(scales[(mode >> 6) & 3]);
  for (int lane = 0; lane < 32; lane += 8) {
    auto x = half_input<false>(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane)),
                               a_shift, mode);
    auto value = unary_value<Op, true>(x);
    if (mode & GOC_ALU_OMOD_HALF)
      value = _mm256_mul_ps(value, scale);
    if (mode & GOC_ALU_CLAMP)
      value = _mm256_min_ps(_mm256_max_ps(value, _mm256_setzero_ps()), _mm256_set1_ps(1));
    auto result = _mm256_sll_epi32(half_narrow<false>(value, saturate), _mm_cvtsi32_si128(d_shift));
    auto old = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d + lane));
    result = _mm256_or_si256(result, _mm256_and_si256(old, keep));
    auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(mask >> lane)),
                                    _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), active, result);
  }
}

template void half_unary_x86_64_v3<Unary::Trunc>(bool, uint32_t, uint32_t, uint32_t *,
                                                 const uint32_t *);
template void half_unary_x86_64_v3<Unary::Ceil>(bool, uint32_t, uint32_t, uint32_t *,
                                                const uint32_t *);
template void half_unary_x86_64_v3<Unary::Rndne>(bool, uint32_t, uint32_t, uint32_t *,
                                                 const uint32_t *);
template void half_unary_x86_64_v3<Unary::Floor>(bool, uint32_t, uint32_t, uint32_t *,
                                                 const uint32_t *);
template void half_unary_x86_64_v3<Unary::Sqrt>(bool, uint32_t, uint32_t, uint32_t *,
                                                const uint32_t *);
template void half_unary_x86_64_v3<Unary::Rcp>(bool, uint32_t, uint32_t, uint32_t *,
                                               const uint32_t *);
template void half_unary_x86_64_v3<Unary::Rsq>(bool, uint32_t, uint32_t, uint32_t *,
                                               const uint32_t *);
template void half_unary_x86_64_v3<Unary::Fract>(bool, uint32_t, uint32_t, uint32_t *,
                                                 const uint32_t *);
template void half_unary_x86_64_v3<Unary::FrexpMant>(bool, uint32_t, uint32_t, uint32_t *,
                                                     const uint32_t *);

} // namespace goc

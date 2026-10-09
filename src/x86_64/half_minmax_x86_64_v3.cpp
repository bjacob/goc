// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "half_minmax.h"
#include "x86_64/half_x86_64_v3.h"
#include "x86_64/minmax_x86_64_v3.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {

template <bool FirstMaximum, bool SecondMaximum, bool Propagate, bool Median>
void half_minmax3_x86_64_v3(bool saturate, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                            const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  int a_shift = mode & GOC_ALU_HIGH_A ? 16 : 0;
  int b_shift = mode & GOC_ALU_HIGH_B ? 16 : 0;
  int c_shift = mode & GOC_ALU_HIGH_C ? 16 : 0;
  int d_shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  auto keep = _mm256_set1_epi32(d_shift ? 65535 : -65536);
  const float scales[] = {1, 2, 4, 0.5f};
  const auto scale = _mm256_set1_ps(scales[(mode >> 6) & 3]);
  for (int lane = 0; lane < 32; lane += 8) {
    auto x = half_input<false>(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane)),
                               a_shift, mode);
    auto y = half_input<false>(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane)),
                               b_shift, mode >> 1);
    auto z = half_input<false>(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(c + lane)),
                               c_shift, mode >> 2);
    auto value = minmax3_value<FirstMaximum, SecondMaximum, Propagate, Median, true>(x, y, z);
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

template void half_minmax3_x86_64_v3<false, false, false, false>(bool, uint32_t, uint32_t,
                                                                 uint32_t *, const uint32_t *,
                                                                 const uint32_t *,
                                                                 const uint32_t *);
template void half_minmax3_x86_64_v3<true, true, false, false>(bool, uint32_t, uint32_t, uint32_t *,
                                                               const uint32_t *, const uint32_t *,
                                                               const uint32_t *);
template void half_minmax3_x86_64_v3<false, true, false, false>(bool, uint32_t, uint32_t,
                                                                uint32_t *, const uint32_t *,
                                                                const uint32_t *, const uint32_t *);
template void half_minmax3_x86_64_v3<true, false, false, false>(bool, uint32_t, uint32_t,
                                                                uint32_t *, const uint32_t *,
                                                                const uint32_t *, const uint32_t *);
template void half_minmax3_x86_64_v3<false, false, true, false>(bool, uint32_t, uint32_t,
                                                                uint32_t *, const uint32_t *,
                                                                const uint32_t *, const uint32_t *);
template void half_minmax3_x86_64_v3<true, true, true, false>(bool, uint32_t, uint32_t, uint32_t *,
                                                              const uint32_t *, const uint32_t *,
                                                              const uint32_t *);
template void half_minmax3_x86_64_v3<false, true, true, false>(bool, uint32_t, uint32_t, uint32_t *,
                                                               const uint32_t *, const uint32_t *,
                                                               const uint32_t *);
template void half_minmax3_x86_64_v3<true, false, true, false>(bool, uint32_t, uint32_t, uint32_t *,
                                                               const uint32_t *, const uint32_t *,
                                                               const uint32_t *);
template void half_minmax3_x86_64_v3<false, false, false, true>(bool, uint32_t, uint32_t,
                                                                uint32_t *, const uint32_t *,
                                                                const uint32_t *, const uint32_t *);

} // namespace goc

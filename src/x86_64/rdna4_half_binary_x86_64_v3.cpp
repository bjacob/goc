// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_binary.h"
#include "rdna4_half_binary.h"
#include "x86_64/rdna4_half_x86_64_v3.h"
#include "x86_64/rdna4_minmax_x86_64_v3.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {

template <Binary Op>
void half_binary_x86_64_v3(bool saturate, uint32_t mask, uint32_t mode, uint32_t *d,
                           const uint32_t *a, const uint32_t *b) {
  int a_shift = mode & GOC_ALU_HIGH_A ? 16 : 0;
  int b_shift = mode & GOC_ALU_HIGH_B ? 16 : 0;
  int d_shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  auto keep = _mm256_set1_epi32(d_shift ? 65535 : -65536);
  const float scales[] = {1, 2, 4, 0.5f};
  const auto scale = _mm256_set1_ps(scales[(mode >> 6) & 3]);
  for (int lane = 0; lane < 32; lane += 8) {
    auto x = half_input<false>(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane)),
                               a_shift, mode);
    auto y = half_input<false>(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane)),
                               b_shift, mode >> 1);
    __m256 value;
    if constexpr (Op == Binary::Add)
      value = _mm256_add_ps(x, y);
    if constexpr (Op == Binary::Sub)
      value = _mm256_sub_ps(x, y);
    if constexpr (Op == Binary::Subrev)
      value = _mm256_sub_ps(y, x);
    if constexpr (Op == Binary::Mul)
      value = _mm256_mul_ps(x, y);
    if constexpr (Op == Binary::MinNum)
      value = minmax<false, false>(x, y);
    if constexpr (Op == Binary::MaxNum)
      value = minmax<true, false>(x, y);
    if constexpr (Op == Binary::Minimum)
      value = minmax<false, true>(x, y);
    if constexpr (Op == Binary::Maximum)
      value = minmax<true, true>(x, y);
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

template void half_binary_x86_64_v3<Binary::Add>(bool, uint32_t, uint32_t, uint32_t *,
                                                 const uint32_t *, const uint32_t *);
template void half_binary_x86_64_v3<Binary::Sub>(bool, uint32_t, uint32_t, uint32_t *,
                                                 const uint32_t *, const uint32_t *);
template void half_binary_x86_64_v3<Binary::Subrev>(bool, uint32_t, uint32_t, uint32_t *,
                                                    const uint32_t *, const uint32_t *);
template void half_binary_x86_64_v3<Binary::Mul>(bool, uint32_t, uint32_t, uint32_t *,
                                                 const uint32_t *, const uint32_t *);
template void half_binary_x86_64_v3<Binary::MinNum>(bool, uint32_t, uint32_t, uint32_t *,
                                                    const uint32_t *, const uint32_t *);
template void half_binary_x86_64_v3<Binary::MaxNum>(bool, uint32_t, uint32_t, uint32_t *,
                                                    const uint32_t *, const uint32_t *);
template void half_binary_x86_64_v3<Binary::Minimum>(bool, uint32_t, uint32_t, uint32_t *,
                                                     const uint32_t *, const uint32_t *);
template void half_binary_x86_64_v3<Binary::Maximum>(bool, uint32_t, uint32_t, uint32_t *,
                                                     const uint32_t *, const uint32_t *);

} // namespace goc

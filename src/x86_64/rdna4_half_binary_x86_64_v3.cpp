// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_binary.h"
#include "rdna4_half_binary.h"
#include "rdna4_packed_alu.h"
#include "x86_64/rdna4_half_x86_64_v3.h"
#include "x86_64/rdna4_minmax_x86_64_v3.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {

template <Binary Op, bool Packed>
void half_binary_x86_64_v3(bool saturate, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                           const uint32_t *a, const uint32_t *b) {
  uint32_t modes[] = {Packed ? packed_half_mode(mode, false) : mode, packed_half_mode(mode, true)};
  int d_shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  auto keep = _mm256_set1_epi32(d_shift ? 65535 : -65536);
  const float scales[] = {1, 2, 4, 0.5f};
  for (int lane = 0; lane < 32; lane += 8) {
    auto a_words = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    auto b_words = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
    auto result = _mm256_setzero_si256();
    for (int half = 0; half < (Packed ? 2 : 1); ++half) {
      uint32_t m = modes[half];
      auto x = half_input<false>(a_words, m & GOC_ALU_HIGH_A ? 16 : 0, m);
      auto y = half_input<false>(b_words, m & GOC_ALU_HIGH_B ? 16 : 0, m >> 1);
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
      if (m & GOC_ALU_OMOD_HALF) {
        value = prepare_omod_f16(value, m);
        value = _mm256_mul_ps(value, _mm256_set1_ps(scales[(m >> 6) & 3]));
      }
      if (m & GOC_ALU_CLAMP)
        value = _mm256_min_ps(_mm256_max_ps(value, _mm256_setzero_ps()), _mm256_set1_ps(1));
      auto encoded = half_narrow<false>(value, saturate);
      if constexpr (Packed)
        result = _mm256_or_si256(result, _mm256_sll_epi32(encoded, _mm_cvtsi32_si128(16 * half)));
      else
        result = _mm256_sll_epi32(encoded, _mm_cvtsi32_si128(d_shift));
    }
    if constexpr (!Packed) {
      auto old = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d + lane));
      result = _mm256_or_si256(result, _mm256_and_si256(old, keep));
    }
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                            _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask, result);
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

template void half_binary_x86_64_v3<Binary::Add, true>(bool, uint32_t, uint32_t, uint32_t *,
                                                       const uint32_t *, const uint32_t *);
template void half_binary_x86_64_v3<Binary::Mul, true>(bool, uint32_t, uint32_t, uint32_t *,
                                                       const uint32_t *, const uint32_t *);
template void half_binary_x86_64_v3<Binary::MinNum, true>(bool, uint32_t, uint32_t, uint32_t *,
                                                          const uint32_t *, const uint32_t *);
template void half_binary_x86_64_v3<Binary::MaxNum, true>(bool, uint32_t, uint32_t, uint32_t *,
                                                          const uint32_t *, const uint32_t *);
template void half_binary_x86_64_v3<Binary::Minimum, true>(bool, uint32_t, uint32_t, uint32_t *,
                                                           const uint32_t *, const uint32_t *);
template void half_binary_x86_64_v3<Binary::Maximum, true>(bool, uint32_t, uint32_t, uint32_t *,
                                                           const uint32_t *, const uint32_t *);

} // namespace goc

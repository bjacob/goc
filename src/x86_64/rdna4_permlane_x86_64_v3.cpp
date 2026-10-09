// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_permlane.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {
template <bool Cross, bool Var>
void permlane_x86_64_v3(uint32_t mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                        const uint32_t *b, uint32_t lo, uint32_t hi) {
  __m256i results[4];
  auto ones = _mm256_set1_epi32(1);
  auto bits = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
  auto shifts = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
  for (unsigned lane = 0; lane < 32; lane += 8) {
    __m256i index;
    if constexpr (Var)
      index = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
    else {
      auto packed = _mm256_set1_epi32(int(lane & 8 ? hi : lo));
      index = _mm256_srlv_epi32(packed, shifts);
    }
    index = _mm256_and_si256(index, _mm256_set1_epi32(15));
    unsigned base = (lane & 16) ^ (Cross ? 16 : 0);
    auto low = _mm256_permutevar8x32_epi32(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + base)), index);
    auto high = _mm256_permutevar8x32_epi32(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + base + 8)), index);
    auto value = _mm256_blendv_epi8(low, high, _mm256_cmpgt_epi32(index, _mm256_set1_epi32(7)));
    if (!(mode & GOC_PERMLANE_FI)) {
      auto active =
          _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(int(mask >> base)), index), ones);
      auto fallback = mode & GOC_PERMLANE_BOUND_CTRL
                          ? _mm256_setzero_si256()
                          : _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d + lane));
      value = _mm256_blendv_epi8(fallback, value, _mm256_sub_epi32(_mm256_setzero_si256(), active));
    }
    results[lane / 8] = value;
  }
  // Defer every store until both source rows and all indices have been read.
  for (unsigned lane = 0; lane < 32; lane += 8) {
    auto enabled =
        _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_set1_epi32(int(mask >> lane)), bits), bits);
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), enabled, results[lane / 8]);
  }
}

template void permlane_x86_64_v3<false, false>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                               const uint32_t *, uint32_t, uint32_t);
template void permlane_x86_64_v3<false, true>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                              const uint32_t *, uint32_t, uint32_t);
template void permlane_x86_64_v3<true, false>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                              const uint32_t *, uint32_t, uint32_t);
template void permlane_x86_64_v3<true, true>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                             const uint32_t *, uint32_t, uint32_t);

} // namespace goc

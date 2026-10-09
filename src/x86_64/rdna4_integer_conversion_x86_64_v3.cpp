// SPDX-License-Identifier: MIT

#include "rdna4_integer_conversion.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {
namespace {

template <bool Unsigned> __m256i narrow(__m256i raw) {
  if constexpr (Unsigned)
    return _mm256_min_epu32(raw, _mm256_set1_epi32(65535));
  else
    return _mm256_and_si256(_mm256_min_epi32(_mm256_max_epi32(raw, _mm256_set1_epi32(-32768)),
                                             _mm256_set1_epi32(32767)),
                            _mm256_set1_epi32(65535));
}

} // namespace

template <bool Unsigned>
void integer_conversion_x86_64_v3(uint32_t mask, uint32_t *d, const uint32_t *a,
                                  const uint32_t *b) {
  for (unsigned lane = 0; lane < 32; lane += 8) {
    auto va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    auto vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
    auto result =
        _mm256_or_si256(narrow<Unsigned>(va), _mm256_slli_epi32(narrow<Unsigned>(vb), 16));
    auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(mask >> lane)),
                                    _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), active, result);
  }
}

template void integer_conversion_x86_64_v3<false>(uint32_t, uint32_t *, const uint32_t *,
                                                  const uint32_t *);
template void integer_conversion_x86_64_v3<true>(uint32_t, uint32_t *, const uint32_t *,
                                                 const uint32_t *);

} // namespace goc

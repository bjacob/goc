// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_packed_mad.h"
#include "x86_64/rdna4_packed_integer_x86_64_v3.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {
namespace {

template <bool Signed, bool High> __m256i widen(__m256i value) {
  if constexpr (Signed)
    return _mm256_srai_epi32(High ? value : _mm256_slli_epi32(value, 16), 16);
  else
    return High ? _mm256_srli_epi32(value, 16) : _mm256_and_si256(value, _mm256_set1_epi32(65535));
}

template <bool Signed, bool High> __m256i saturated(__m256i a, __m256i b, __m256i c) {
  // Even the largest unsigned result, 65535 * 65535 + 65535, fits uint32_t.
  auto value = _mm256_add_epi32(_mm256_mullo_epi32(widen<Signed, High>(a), widen<Signed, High>(b)),
                                widen<Signed, High>(c));
  if constexpr (Signed)
    return _mm256_min_epi32(_mm256_max_epi32(value, _mm256_set1_epi32(-32768)),
                            _mm256_set1_epi32(32767));
  else
    return _mm256_min_epu32(value, _mm256_set1_epi32(65535));
}

} // namespace

template <bool Signed>
void packed_mad_x86_64_v3(uint32_t mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                          const uint32_t *b, const uint32_t *c) {
  auto sa = packed_integer_selection(mode & GOC_PK_LO_A_HIGH, mode & GOC_PK_HI_A_LOW);
  auto sb = packed_integer_selection(mode & GOC_PK_LO_B_HIGH, mode & GOC_PK_HI_B_LOW);
  auto sc = packed_integer_selection(mode & GOC_PK_LO_C_HIGH, mode & GOC_PK_HI_C_LOW);
  for (int lane = 0; lane < 32; lane += 8) {
    auto x =
        _mm256_shuffle_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane)), sa);
    auto y =
        _mm256_shuffle_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane)), sb);
    auto z =
        _mm256_shuffle_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(c + lane)), sc);
    __m256i result;
    if (mode & GOC_PK_CLAMP) {
      auto low = saturated<Signed, false>(x, y, z), high = saturated<Signed, true>(x, y, z);
      result = _mm256_or_si256(_mm256_and_si256(low, _mm256_set1_epi32(65535)),
                               _mm256_slli_epi32(high, 16));
    } else {
      result = _mm256_add_epi16(_mm256_mullo_epi16(x, y), z);
    }
    auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(mask >> lane)),
                                    _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), active, result);
  }
}

template void packed_mad_x86_64_v3<false>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                          const uint32_t *, const uint32_t *);
template void packed_mad_x86_64_v3<true>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                         const uint32_t *, const uint32_t *);

} // namespace goc

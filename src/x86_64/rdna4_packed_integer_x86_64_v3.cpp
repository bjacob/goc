// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_packed_integer.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {
namespace {

// Byte indices select either half independently within every original VGPR word.
__m256i selection(bool low_high, bool high_low) {
  uint32_t low = low_high ? 0x0302 : 0x0100;
  uint32_t high = high_low ? 0x0100 : 0x0302;
  return _mm256_add_epi8(_mm256_set1_epi32(int(low | (high << 16))),
                         _mm256_setr_epi32(0, 0x04040404, 0x08080808, 0x0c0c0c0c, 0, 0x04040404,
                                           0x08080808, 0x0c0c0c0c));
}

} // namespace

template <PackedInteger Op, bool Signed>
void packed_integer_x86_64_v3(uint32_t mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                              const uint32_t *b) {
  auto sa = selection(mode & GOC_PK_LO_A_HIGH, mode & GOC_PK_HI_A_LOW);
  auto sb = selection(mode & GOC_PK_LO_B_HIGH, mode & GOC_PK_HI_B_LOW);
  for (int lane = 0; lane < 32; lane += 8) {
    auto x =
        _mm256_shuffle_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane)), sa);
    auto y =
        _mm256_shuffle_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane)), sb);
    __m256i result;
    if constexpr (Op == PackedInteger::Add) {
      result = mode & GOC_PK_CLAMP ? (Signed ? _mm256_adds_epi16(x, y) : _mm256_adds_epu16(x, y))
                                   : _mm256_add_epi16(x, y);
    }
    if constexpr (Op == PackedInteger::Sub) {
      result = mode & GOC_PK_CLAMP ? (Signed ? _mm256_subs_epi16(x, y) : _mm256_subs_epu16(x, y))
                                   : _mm256_sub_epi16(x, y);
    }
    if constexpr (Op == PackedInteger::Min)
      result = Signed ? _mm256_min_epi16(x, y) : _mm256_min_epu16(x, y);
    if constexpr (Op == PackedInteger::Max)
      result = Signed ? _mm256_max_epi16(x, y) : _mm256_max_epu16(x, y);
    if constexpr (Op == PackedInteger::Mul)
      result = _mm256_mullo_epi16(x, y);
    auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(mask >> lane)),
                                    _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), active, result);
  }
}

template void packed_integer_x86_64_v3<PackedInteger::Add, true>(uint32_t, uint32_t, uint32_t *,
                                                                 const uint32_t *,
                                                                 const uint32_t *);
template void packed_integer_x86_64_v3<PackedInteger::Sub, true>(uint32_t, uint32_t, uint32_t *,
                                                                 const uint32_t *,
                                                                 const uint32_t *);
template void packed_integer_x86_64_v3<PackedInteger::Add, false>(uint32_t, uint32_t, uint32_t *,
                                                                  const uint32_t *,
                                                                  const uint32_t *);
template void packed_integer_x86_64_v3<PackedInteger::Sub, false>(uint32_t, uint32_t, uint32_t *,
                                                                  const uint32_t *,
                                                                  const uint32_t *);
template void packed_integer_x86_64_v3<PackedInteger::Min, true>(uint32_t, uint32_t, uint32_t *,
                                                                 const uint32_t *,
                                                                 const uint32_t *);
template void packed_integer_x86_64_v3<PackedInteger::Max, true>(uint32_t, uint32_t, uint32_t *,
                                                                 const uint32_t *,
                                                                 const uint32_t *);
template void packed_integer_x86_64_v3<PackedInteger::Min, false>(uint32_t, uint32_t, uint32_t *,
                                                                  const uint32_t *,
                                                                  const uint32_t *);
template void packed_integer_x86_64_v3<PackedInteger::Max, false>(uint32_t, uint32_t, uint32_t *,
                                                                  const uint32_t *,
                                                                  const uint32_t *);
template void packed_integer_x86_64_v3<PackedInteger::Mul, false>(uint32_t, uint32_t, uint32_t *,
                                                                  const uint32_t *,
                                                                  const uint32_t *);

} // namespace goc

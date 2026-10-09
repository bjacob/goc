// SPDX-License-Identifier: MIT

#include "rdna4_bitfield.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {

template <Bitfield Op>
void bitfield_x86_64_v3(uint32_t mask, uint32_t *d, const uint32_t *a, const uint32_t *b,
                        const uint32_t *c) {
  static_assert(Op != Bitfield::Insert);
  for (int lane = 0; lane < 32; lane += 8) {
    __m256i x;
    x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    __m256i y;
    if constexpr (Op != Bitfield::Reverse)
      y = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
    __m256i z;
    if constexpr (Op != Bitfield::Reverse && Op != Bitfield::Mask)
      z = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c + lane));
    __m256i result;
    if constexpr (Op == Bitfield::ExtractUnsigned || Op == Bitfield::ExtractSigned) {
      auto offset = _mm256_and_si256(y, _mm256_set1_epi32(31)),
           width = _mm256_and_si256(z, _mm256_set1_epi32(31));
      auto power = _mm256_sllv_epi32(_mm256_set1_epi32(1), width);
      auto field_mask = _mm256_sub_epi32(power, _mm256_set1_epi32(1));
      if constexpr (Op == Bitfield::ExtractUnsigned)
        result = _mm256_and_si256(_mm256_srlv_epi32(x, offset), field_mask);
      else {
        auto field = _mm256_and_si256(_mm256_srav_epi32(x, offset), field_mask);
        auto sign = _mm256_srli_epi32(power, 1);
        result = _mm256_sub_epi32(_mm256_xor_si256(field, sign), sign);
      }
    }
    if constexpr (Op == Bitfield::AlignBit || Op == Bitfield::AlignByte) {
      auto shift = _mm256_and_si256(z, _mm256_set1_epi32(Op == Bitfield::AlignBit ? 31 : 3));
      if constexpr (Op == Bitfield::AlignByte)
        shift = _mm256_slli_epi32(shift, 3);
      auto inverse = _mm256_sub_epi32(_mm256_set1_epi32(32), shift);
      result = _mm256_or_si256(_mm256_srlv_epi32(y, shift), _mm256_sllv_epi32(x, inverse));
    }
    if constexpr (Op == Bitfield::Mask)
      result = _mm256_sllv_epi32(
          _mm256_sub_epi32(
              _mm256_sllv_epi32(_mm256_set1_epi32(1), _mm256_and_si256(x, _mm256_set1_epi32(31))),
              _mm256_set1_epi32(1)),
          _mm256_and_si256(y, _mm256_set1_epi32(31)));
    if constexpr (Op == Bitfield::Reverse) {
      // Reverse each nibble with a byte lookup, then reverse the four bytes.
      auto table = _mm256_broadcastsi128_si256(
          _mm_setr_epi8(0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15));
      auto nibble = _mm256_set1_epi8(15);
      auto low = _mm256_shuffle_epi8(table, _mm256_and_si256(x, nibble));
      auto high = _mm256_shuffle_epi8(table, _mm256_and_si256(_mm256_srli_epi16(x, 4), nibble));
      result = _mm256_shuffle_epi8(_mm256_or_si256(_mm256_slli_epi16(low, 4), high),
                                   _mm256_broadcastsi128_si256(_mm_setr_epi8(
                                       3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12)));
    }
    auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(mask >> lane)),
                                    _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), active, result);
  }
}

template void bitfield_x86_64_v3<Bitfield::ExtractUnsigned>(uint32_t mask, uint32_t *d,
                                                            const uint32_t *a, const uint32_t *b,
                                                            const uint32_t *c);
template void bitfield_x86_64_v3<Bitfield::ExtractSigned>(uint32_t mask, uint32_t *d,
                                                          const uint32_t *a, const uint32_t *b,
                                                          const uint32_t *c);
template void bitfield_x86_64_v3<Bitfield::Mask>(uint32_t mask, uint32_t *d, const uint32_t *a,
                                                 const uint32_t *b, const uint32_t *c);
template void bitfield_x86_64_v3<Bitfield::Reverse>(uint32_t mask, uint32_t *d, const uint32_t *a,
                                                    const uint32_t *b, const uint32_t *c);

template void bitfield_x86_64_v3<Bitfield::AlignBit>(uint32_t, uint32_t *, const uint32_t *,
                                                     const uint32_t *, const uint32_t *);
template void bitfield_x86_64_v3<Bitfield::AlignByte>(uint32_t, uint32_t *, const uint32_t *,
                                                      const uint32_t *, const uint32_t *);

} // namespace goc

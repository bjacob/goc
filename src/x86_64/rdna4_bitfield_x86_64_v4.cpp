// SPDX-License-Identifier: MIT

#include "rdna4_bitfield.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {

template <Bitfield Op>
void bitfield_x86_64_v4(uint32_t mask, uint32_t *d, const uint32_t *a, const uint32_t *b,
                        const uint32_t *c) {
  for (int lane = 0; lane < 32; lane += 16) {
    __m512i x;
    x = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + lane));
    __m512i y;
    if constexpr (Op != Bitfield::Reverse)
      y = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(b + lane));
    __m512i z;
    if constexpr (Op != Bitfield::Reverse && Op != Bitfield::Mask)
      z = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(c + lane));
    __m512i result;
    if constexpr (Op == Bitfield::ExtractUnsigned || Op == Bitfield::ExtractSigned) {
      auto offset = _mm512_and_si512(y, _mm512_set1_epi32(31)),
           width = _mm512_and_si512(z, _mm512_set1_epi32(31));
      auto power = _mm512_sllv_epi32(_mm512_set1_epi32(1), width);
      auto field_mask = _mm512_sub_epi32(power, _mm512_set1_epi32(1));
      if constexpr (Op == Bitfield::ExtractUnsigned)
        result = _mm512_and_si512(_mm512_srlv_epi32(x, offset), field_mask);
      else {
        auto field = _mm512_and_si512(_mm512_srav_epi32(x, offset), field_mask);
        auto sign = _mm512_srli_epi32(power, 1);
        result = _mm512_sub_epi32(_mm512_xor_si512(field, sign), sign);
      }
    }
    if constexpr (Op == Bitfield::Insert)
      result = _mm512_ternarylogic_epi32(x, y, z, 0xca);
    if constexpr (Op == Bitfield::AlignBit || Op == Bitfield::AlignByte) {
      auto shift = _mm512_and_si512(z, _mm512_set1_epi32(Op == Bitfield::AlignBit ? 31 : 3));
      if constexpr (Op == Bitfield::AlignByte)
        shift = _mm512_slli_epi32(shift, 3);
      auto inverse = _mm512_sub_epi32(_mm512_set1_epi32(32), shift);
      result = _mm512_or_si512(_mm512_srlv_epi32(y, shift), _mm512_sllv_epi32(x, inverse));
    }
    if constexpr (Op == Bitfield::Mask)
      result = _mm512_sllv_epi32(
          _mm512_sub_epi32(
              _mm512_sllv_epi32(_mm512_set1_epi32(1), _mm512_and_si512(x, _mm512_set1_epi32(31))),
              _mm512_set1_epi32(1)),
          _mm512_and_si512(y, _mm512_set1_epi32(31)));
    if constexpr (Op == Bitfield::Reverse) {
      // Reverse each nibble with a byte lookup, then reverse the four bytes.
      auto table = _mm512_broadcast_i32x4(
          _mm_setr_epi8(0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15));
      auto nibble = _mm512_set1_epi8(15);
      auto low = _mm512_shuffle_epi8(table, _mm512_and_si512(x, nibble));
      auto high = _mm512_shuffle_epi8(table, _mm512_and_si512(_mm512_srli_epi16(x, 4), nibble));
      result = _mm512_shuffle_epi8(_mm512_or_si512(_mm512_slli_epi16(low, 4), high),
                                   _mm512_broadcast_i32x4(_mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11,
                                                                        10, 9, 8, 15, 14, 13, 12)));
    }
    _mm512_mask_storeu_epi32(d + lane, __mmask16(mask >> lane), result);
  }
}

template void bitfield_x86_64_v4<Bitfield::ExtractUnsigned>(uint32_t mask, uint32_t *d,
                                                            const uint32_t *a, const uint32_t *b,
                                                            const uint32_t *c);
template void bitfield_x86_64_v4<Bitfield::ExtractSigned>(uint32_t mask, uint32_t *d,
                                                          const uint32_t *a, const uint32_t *b,
                                                          const uint32_t *c);
template void bitfield_x86_64_v4<Bitfield::Insert>(uint32_t mask, uint32_t *d, const uint32_t *a,
                                                   const uint32_t *b, const uint32_t *c);
template void bitfield_x86_64_v4<Bitfield::Mask>(uint32_t mask, uint32_t *d, const uint32_t *a,
                                                 const uint32_t *b, const uint32_t *c);
template void bitfield_x86_64_v4<Bitfield::Reverse>(uint32_t mask, uint32_t *d, const uint32_t *a,
                                                    const uint32_t *b, const uint32_t *c);

template void bitfield_x86_64_v4<Bitfield::AlignBit>(uint32_t, uint32_t *, const uint32_t *,
                                                     const uint32_t *, const uint32_t *);
template void bitfield_x86_64_v4<Bitfield::AlignByte>(uint32_t, uint32_t *, const uint32_t *,
                                                      const uint32_t *, const uint32_t *);

} // namespace goc

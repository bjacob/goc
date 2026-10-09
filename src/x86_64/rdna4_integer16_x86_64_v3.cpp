// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_integer16.h"
#include "x86_64/rdna4_packed_integer_x86_64_v3.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {

template <Integer16 Op, bool Signed, bool Packed>
void integer16_x86_64_v3(uint32_t mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                         const uint32_t *b) {
  const int d_shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  if constexpr (!Packed)
    mode = (mode & GOC_ALU_HIGH_A ? GOC_PK_LO_A_HIGH : 0) |
           (mode & GOC_ALU_HIGH_B ? GOC_PK_LO_B_HIGH : 0) |
           (mode & GOC_ALU_CLAMP ? GOC_PK_CLAMP : 0);
  auto sa = packed_integer_selection(mode & GOC_PK_LO_A_HIGH, mode & GOC_PK_HI_A_LOW);
  auto sb = packed_integer_selection(mode & GOC_PK_LO_B_HIGH, mode & GOC_PK_HI_B_LOW);
  for (int lane = 0; lane < 32; lane += 8) {
    auto x =
        _mm256_shuffle_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane)), sa);
    auto y =
        _mm256_shuffle_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane)), sb);
    __m256i result;
    if constexpr (Op == Integer16::Add) {
      result = mode & GOC_PK_CLAMP ? (Signed ? _mm256_adds_epi16(x, y) : _mm256_adds_epu16(x, y))
                                   : _mm256_add_epi16(x, y);
    }
    if constexpr (Op == Integer16::Sub) {
      result = mode & GOC_PK_CLAMP ? (Signed ? _mm256_subs_epi16(x, y) : _mm256_subs_epu16(x, y))
                                   : _mm256_sub_epi16(x, y);
    }
    if constexpr (Op == Integer16::Min)
      result = Signed ? _mm256_min_epi16(x, y) : _mm256_min_epu16(x, y);
    if constexpr (Op == Integer16::Max)
      result = Signed ? _mm256_max_epi16(x, y) : _mm256_max_epu16(x, y);
    if constexpr (Op == Integer16::Mul)
      result = _mm256_mullo_epi16(x, y);
    if constexpr (Op == Integer16::ShiftLeft || Op == Integer16::ShiftRight) {
      const auto halves = _mm256_set1_epi32(65535);
      auto count_lo = _mm256_and_si256(x, _mm256_set1_epi32(15));
      auto count_hi = _mm256_and_si256(_mm256_srli_epi32(x, 16), _mm256_set1_epi32(15));
      __m256i lo, hi;
      if constexpr (Op == Integer16::ShiftLeft) {
        lo = _mm256_sllv_epi32(y, count_lo);
        hi = _mm256_sllv_epi32(_mm256_srli_epi32(y, 16), count_hi);
      } else if constexpr (Signed) {
        lo = _mm256_srav_epi32(_mm256_srai_epi32(_mm256_slli_epi32(y, 16), 16), count_lo);
        hi = _mm256_srav_epi32(_mm256_srai_epi32(y, 16), count_hi);
      } else {
        lo = _mm256_srlv_epi32(_mm256_and_si256(y, halves), count_lo);
        hi = _mm256_srlv_epi32(_mm256_srli_epi32(y, 16), count_hi);
      }
      result = _mm256_or_si256(_mm256_and_si256(lo, halves), _mm256_slli_epi32(hi, 16));
    }
    if constexpr (!Packed) {
      auto output_mask = _mm256_set1_epi32(int(UINT32_C(65535) << d_shift));
      auto original = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d + lane));
      auto selected = _mm256_sllv_epi32(result, _mm256_set1_epi32(d_shift));
      result = _mm256_or_si256(_mm256_and_si256(selected, output_mask),
                               _mm256_andnot_si256(output_mask, original));
    }
    auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(mask >> lane)),
                                    _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), active, result);
  }
}

template void integer16_x86_64_v3<Integer16::Add, true>(uint32_t, uint32_t, uint32_t *,
                                                        const uint32_t *, const uint32_t *);
template void integer16_x86_64_v3<Integer16::Sub, true>(uint32_t, uint32_t, uint32_t *,
                                                        const uint32_t *, const uint32_t *);
template void integer16_x86_64_v3<Integer16::Add, false>(uint32_t, uint32_t, uint32_t *,
                                                         const uint32_t *, const uint32_t *);
template void integer16_x86_64_v3<Integer16::Sub, false>(uint32_t, uint32_t, uint32_t *,
                                                         const uint32_t *, const uint32_t *);
template void integer16_x86_64_v3<Integer16::Min, true>(uint32_t, uint32_t, uint32_t *,
                                                        const uint32_t *, const uint32_t *);
template void integer16_x86_64_v3<Integer16::Max, true>(uint32_t, uint32_t, uint32_t *,
                                                        const uint32_t *, const uint32_t *);
template void integer16_x86_64_v3<Integer16::Min, false>(uint32_t, uint32_t, uint32_t *,
                                                         const uint32_t *, const uint32_t *);
template void integer16_x86_64_v3<Integer16::Max, false>(uint32_t, uint32_t, uint32_t *,
                                                         const uint32_t *, const uint32_t *);
template void integer16_x86_64_v3<Integer16::Mul, false>(uint32_t, uint32_t, uint32_t *,
                                                         const uint32_t *, const uint32_t *);

template void integer16_x86_64_v3<Integer16::ShiftLeft, false>(uint32_t, uint32_t, uint32_t *,
                                                               const uint32_t *, const uint32_t *);
template void integer16_x86_64_v3<Integer16::ShiftRight, false>(uint32_t, uint32_t, uint32_t *,
                                                                const uint32_t *, const uint32_t *);
template void integer16_x86_64_v3<Integer16::ShiftRight, true>(uint32_t, uint32_t, uint32_t *,
                                                               const uint32_t *, const uint32_t *);

template void integer16_x86_64_v3<Integer16::Add, true, false>(uint32_t, uint32_t, uint32_t *,
                                                               const uint32_t *, const uint32_t *);
template void integer16_x86_64_v3<Integer16::Sub, true, false>(uint32_t, uint32_t, uint32_t *,
                                                               const uint32_t *, const uint32_t *);
template void integer16_x86_64_v3<Integer16::Add, false, false>(uint32_t, uint32_t, uint32_t *,
                                                                const uint32_t *, const uint32_t *);
template void integer16_x86_64_v3<Integer16::Sub, false, false>(uint32_t, uint32_t, uint32_t *,
                                                                const uint32_t *, const uint32_t *);
template void integer16_x86_64_v3<Integer16::Min, true, false>(uint32_t, uint32_t, uint32_t *,
                                                               const uint32_t *, const uint32_t *);
template void integer16_x86_64_v3<Integer16::Max, true, false>(uint32_t, uint32_t, uint32_t *,
                                                               const uint32_t *, const uint32_t *);
template void integer16_x86_64_v3<Integer16::Min, false, false>(uint32_t, uint32_t, uint32_t *,
                                                                const uint32_t *, const uint32_t *);
template void integer16_x86_64_v3<Integer16::Max, false, false>(uint32_t, uint32_t, uint32_t *,
                                                                const uint32_t *, const uint32_t *);
template void integer16_x86_64_v3<Integer16::Mul, false, false>(uint32_t, uint32_t, uint32_t *,
                                                                const uint32_t *, const uint32_t *);
template void integer16_x86_64_v3<Integer16::ShiftLeft, false, false>(uint32_t, uint32_t,
                                                                      uint32_t *, const uint32_t *,
                                                                      const uint32_t *);
template void integer16_x86_64_v3<Integer16::ShiftRight, false, false>(uint32_t, uint32_t,
                                                                       uint32_t *, const uint32_t *,
                                                                       const uint32_t *);
template void integer16_x86_64_v3<Integer16::ShiftRight, true, false>(uint32_t, uint32_t,
                                                                      uint32_t *, const uint32_t *,
                                                                      const uint32_t *);

} // namespace goc

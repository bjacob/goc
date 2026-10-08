// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_simd.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {
namespace {

// Widen pairs to signed 16-bit factors, including unsigned bytes up to 255.
// The full dot fits int32; only the final addition of C can overflow.
template <int Bits, bool Unsigned>
void dot(uint32_t mask, uint32_t modifiers, uint32_t *d, const uint32_t *a, const uint32_t *b,
         const uint32_t *c) {
  const auto sign_a =
      _mm256_set1_epi32(!Unsigned && (modifiers & GOC_DOT_SIGNED_A) ? 1 << (Bits - 1) : 0);
  const auto sign_b =
      _mm256_set1_epi32(!Unsigned && (modifiers & GOC_DOT_SIGNED_B) ? 1 << (Bits - 1) : 0);
  const auto fields = _mm256_set1_epi32(((1 << Bits) - 1) * 0x10001);
  const auto bias_a = _mm256_or_si256(sign_a, _mm256_slli_epi32(sign_a, 16));
  const auto bias_b = _mm256_or_si256(sign_b, _mm256_slli_epi32(sign_b, 16));
  for (int lane = 0; lane < 32; lane += 8) {
    auto va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    auto vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
    auto acc = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c + lane));
    auto sum = _mm256_setzero_si256();
    for (int shift = 0; shift < 16; shift += Bits) {
      // Pair corresponding fields in the low and high halfwords.
      auto x = _mm256_and_si256(_mm256_srl_epi32(va, _mm_cvtsi32_si128(shift)), fields);
      auto y = _mm256_and_si256(_mm256_srl_epi32(vb, _mm_cvtsi32_si128(shift)), fields);
      x = _mm256_sub_epi16(_mm256_xor_si256(x, bias_a), bias_a);
      y = _mm256_sub_epi16(_mm256_xor_si256(y, bias_b), bias_b);
      sum = _mm256_add_epi32(sum, _mm256_madd_epi16(x, y));
    }
    auto result = _mm256_add_epi32(acc, sum);
    if (modifiers & GOC_DOT_CLAMP) {
      if constexpr (Unsigned) {
        const auto sign = _mm256_set1_epi32(INT32_MIN);
        auto overflow =
            _mm256_cmpgt_epi32(_mm256_xor_si256(acc, sign), _mm256_xor_si256(result, sign));
        result = _mm256_or_si256(result, overflow);
      } else {
        auto overflow =
            _mm256_and_si256(_mm256_xor_si256(acc, result), _mm256_xor_si256(sum, result));
        auto limit = _mm256_xor_si256(_mm256_srai_epi32(acc, 31), _mm256_set1_epi32(INT32_MAX));
        result = _mm256_blendv_epi8(result, limit, _mm256_srai_epi32(overflow, 31));
      }
    }
    auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(mask >> lane)),
                                    _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), active, result);
  }
}

} // namespace

void integer_dot_x86_64_v3(int bits, bool unsigned_acc, uint32_t mask, uint32_t modifiers,
                           uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  if (bits == 8) {
    if (unsigned_acc)
      dot<8, true>(mask, modifiers, d, a, b, c);
    else
      dot<8, false>(mask, modifiers, d, a, b, c);
  } else {
    if (unsigned_acc)
      dot<4, true>(mask, modifiers, d, a, b, c);
    else
      dot<4, false>(mask, modifiers, d, a, b, c);
  }
}

} // namespace goc

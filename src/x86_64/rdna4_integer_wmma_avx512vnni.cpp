// SPDX-License-Identifier: MIT

#include "rdna4_simd.h"
#include "x86_64/rdna4_integer_wmma.h"

#include <cstring>
#include <immintrin.h>
#include <stdint.h>

namespace {

struct Ops {
  using V = __m512i;
  static const int width = 16;

  static V splat(int x) { return _mm512_set1_epi32(x); }

  static V splat_bits(uint32_t x) {
    int32_t s;
    std::memcpy(&s, &x, sizeof(s));
    return splat(s);
  }

  static V load(const uint32_t *x) { return _mm512_loadu_si512(reinterpret_cast<const V *>(x)); }

  static void store(uint32_t *x, V v) { _mm512_storeu_si512(reinterpret_cast<V *>(x), v); }

  static V add(V a, V b) { return _mm512_add_epi32(a, b); }

  static V sub(V a, V b) { return _mm512_sub_epi32(a, b); }

  static V bit_and(V a, V b) { return _mm512_and_si512(a, b); }

  static V bit_or(V a, V b) { return _mm512_or_si512(a, b); }

  static V bit_xor(V a, V b) { return _mm512_xor_si512(a, b); }

  static V shift_right(V a, int n) { return _mm512_srl_epi32(a, _mm_cvtsi32_si128(n)); }

  static V shift_left16(V a) { return _mm512_slli_epi32(a, 16); }

  static V sign(V a) { return _mm512_srai_epi32(a, 31); }

  static V dot(V sum, V a, V b) { return _mm512_dpwssd_epi32(sum, a, b); }

  static V select_negative(V mask, V yes, V no) {
    return _mm512_mask_mov_epi32(no, _mm512_cmp_epi32_mask(mask, splat(0), _MM_CMPINT_LT), yes);
  }
};

} // namespace

namespace goc {

void integer_wmma_avx512vnni(int bits, int k, uint32_t modifiers, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c) {
  if (bits == 8)
    integer_wmma<8, 16, Ops>(modifiers, d, a, b, c);
  else if (k == 16)
    integer_wmma<4, 16, Ops>(modifiers, d, a, b, c);
  else
    integer_wmma<4, 32, Ops>(modifiers, d, a, b, c);
}

} // namespace goc

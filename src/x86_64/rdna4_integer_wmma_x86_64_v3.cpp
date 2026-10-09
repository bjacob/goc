// SPDX-License-Identifier: MIT

#include "rdna4_simd.h"
#include "x86_64/rdna4_integer_wmma.h"

#include <cstring>
#include <immintrin.h>
#include <stdint.h>

namespace {

struct Ops {
  using V = __m256i;
  static const int width = 8;

  static V splat(int x) { return _mm256_set1_epi32(x); }

  static V splat_bits(uint32_t x) {
    int32_t s;
    std::memcpy(&s, &x, sizeof(s));
    return splat(s);
  }

  static V load(const uint32_t *x) { return _mm256_loadu_si256(reinterpret_cast<const V *>(x)); }

  static void store(uint32_t *x, V v) { _mm256_storeu_si256(reinterpret_cast<V *>(x), v); }

  static V add(V a, V b) { return _mm256_add_epi32(a, b); }

  static V sub(V a, V b) { return _mm256_sub_epi32(a, b); }

  static V bit_and(V a, V b) { return _mm256_and_si256(a, b); }

  static V bit_or(V a, V b) { return _mm256_or_si256(a, b); }

  static V bit_xor(V a, V b) { return _mm256_xor_si256(a, b); }

  static V shift_right(V a, int n) { return _mm256_srl_epi32(a, _mm_cvtsi32_si128(n)); }

  static V shift_left16(V a) { return _mm256_slli_epi32(a, 16); }

  static V sign(V a) { return _mm256_srai_epi32(a, 31); }

  static V dot(V sum, V a, V b) { return _mm256_add_epi32(sum, _mm256_madd_epi16(a, b)); }

  static V select_negative(V exec_mask, V yes, V no) {
    return _mm256_blendv_epi8(no, yes, sign(exec_mask));
  }

  static void masked_store(uint32_t *p, uint32_t exec_mask, V v) {
    if ((exec_mask & 255) == 255) {
      store(p, v);
      return;
    }
    V lane_exec_mask = _mm256_setr_epi32(-int(exec_mask & 1), -int((exec_mask >> 1) & 1),
                                         -int((exec_mask >> 2) & 1), -int((exec_mask >> 3) & 1),
                                         -int((exec_mask >> 4) & 1), -int((exec_mask >> 5) & 1),
                                         -int((exec_mask >> 6) & 1), -int((exec_mask >> 7) & 1));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(p), lane_exec_mask, v);
  }
};

} // namespace

namespace goc {

void integer_wmma_x86_64_v3(int bits, int k, uint32_t exec_mask, uint32_t modifiers,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  if (bits == 8)
    integer_wmma<8, 16, Ops>(exec_mask, modifiers, d, a, b, c);
  else if (k == 16)
    integer_wmma<4, 16, Ops>(exec_mask, modifiers, d, a, b, c);
  else
    integer_wmma<4, 32, Ops>(exec_mask, modifiers, d, a, b, c);
}

} // namespace goc

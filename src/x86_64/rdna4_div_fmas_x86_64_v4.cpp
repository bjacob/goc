// SPDX-License-Identifier: MIT

#include "rdna4_div_fmas.h"
#include "x86_64/rdna4_div_fmas_simd.h"

#include <immintrin.h>
#include <stdint.h>

namespace {

struct Ops {
  using V = __m512i;
  using Mask = __mmask8;
  static constexpr unsigned lanes = 8;

  static V set(uint64_t a) { return _mm512_set1_epi64(int64_t(a)); }

  static V add(V a, V b) { return _mm512_add_epi64(a, b); }

  static V sub(V a, V b) { return _mm512_sub_epi64(a, b); }

  static V band(V a, V b) { return _mm512_and_si512(a, b); }

  static V bor(V a, V b) { return _mm512_or_si512(a, b); }

  static V bxor(V a, V b) { return _mm512_xor_si512(a, b); }

  static V shl(V a, V b) { return _mm512_sllv_epi64(a, b); }

  static V shr(V a, V b) { return _mm512_srlv_epi64(a, b); }

  static V mul32(V a, V b) { return _mm512_mul_epu32(a, b); }

  static Mask eq(V a, V b) { return _mm512_cmpeq_epi64_mask(a, b); }

  static Mask gt(V a, V b) { return _mm512_cmp_epi64_mask(a, b, _MM_CMPINT_GT); }

  static Mask ugt(V a, V b) { return _mm512_cmp_epu64_mask(a, b, _MM_CMPINT_GT); }

  static Mask both(Mask a, Mask b) { return a & b; }

  static Mask either(Mask a, Mask b) { return a | b; }

  static Mask inverse(Mask a) { return Mask(~a); }

  static V select(Mask m, V yes, V no) { return _mm512_mask_blend_epi64(m, no, yes); }

  static Mask lane_mask(uint32_t mask) { return __mmask8(mask); }

  static V load_words(const uint32_t *p) {
    return _mm512_cvtepu32_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(p)));
  }

  static void store_words(uint32_t *p, V value) {
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(p), _mm512_cvtepi64_epi32(value));
  }

  static void masked_words(uint32_t *d, const uint32_t *p, uint32_t mask) {
    _mm256_mask_storeu_epi32(d, __mmask8(mask),
                             _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p)));
  }
};

} // namespace

namespace goc {

template <unsigned Width>
void div_fmas_x86_64_v4(uint32_t mask, uint32_t mode, uint32_t *const *d, const uint32_t *const *a,
                        const uint32_t *const *b, const uint32_t *const *c, uint32_t condition) {
  division_vector_run<Width, Ops>(mask, mode, d, a, b, c, condition);
}

template void div_fmas_x86_64_v4<32>(uint32_t, uint32_t, uint32_t *const *, const uint32_t *const *,
                                     const uint32_t *const *, const uint32_t *const *, uint32_t);
template void div_fmas_x86_64_v4<64>(uint32_t, uint32_t, uint32_t *const *, const uint32_t *const *,
                                     const uint32_t *const *, const uint32_t *const *, uint32_t);

} // namespace goc

// SPDX-License-Identifier: MIT

#include "rdna4_div_fmas.h"
#include "x86_64/rdna4_div_fmas_simd.h"

#include <immintrin.h>
#include <stdint.h>

namespace {

struct Ops {
  using V = __m256i;
  using Mask = __m256i;
  static constexpr unsigned lanes = 4;

  static V set(uint64_t a) { return _mm256_set1_epi64x(int64_t(a)); }

  static V add(V a, V b) { return _mm256_add_epi64(a, b); }

  static V sub(V a, V b) { return _mm256_sub_epi64(a, b); }

  static V band(V a, V b) { return _mm256_and_si256(a, b); }

  static V bor(V a, V b) { return _mm256_or_si256(a, b); }

  static V bxor(V a, V b) { return _mm256_xor_si256(a, b); }

  static V shl(V a, V b) { return _mm256_sllv_epi64(a, b); }

  static V shr(V a, V b) { return _mm256_srlv_epi64(a, b); }

  static V mul32(V a, V b) { return _mm256_mul_epu32(a, b); }

  static Mask eq(V a, V b) { return _mm256_cmpeq_epi64(a, b); }

  static Mask gt(V a, V b) { return _mm256_cmpgt_epi64(a, b); }

  static Mask ugt(V a, V b) { return gt(bxor(a, set(1ULL << 63)), bxor(b, set(1ULL << 63))); }

  static Mask both(Mask a, Mask b) { return band(a, b); }

  static Mask either(Mask a, Mask b) { return bor(a, b); }

  static Mask inverse(Mask a) { return bxor(a, set(UINT64_MAX)); }

  static V select(Mask m, V yes, V no) { return _mm256_blendv_epi8(no, yes, m); }

  static Mask lane_mask(uint32_t mask) {
    return _mm256_setr_epi64x(-int64_t(mask & 1), -int64_t((mask >> 1) & 1),
                              -int64_t((mask >> 2) & 1), -int64_t((mask >> 3) & 1));
  }

  static V load_words(const uint32_t *p) {
    return _mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p)));
  }

  static void store_words(uint32_t *p, V value) {
    auto words = _mm256_permutevar8x32_epi32(value, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(p), _mm256_castsi256_si128(words));
  }

  static void masked_words(uint32_t *d, const uint32_t *p, uint32_t mask) {
    auto active = _mm_setr_epi32(-int(mask & 1), -int((mask >> 1) & 1), -int((mask >> 2) & 1),
                                 -int((mask >> 3) & 1));
    _mm_maskstore_epi32(reinterpret_cast<int *>(d), active,
                        _mm_loadu_si128(reinterpret_cast<const __m128i *>(p)));
  }
};

} // namespace

namespace goc {

template <unsigned Width>
void div_fmas_x86_64_v3(uint32_t mask, uint32_t mode, uint32_t *const *d, const uint32_t *const *a,
                        const uint32_t *const *b, const uint32_t *const *c, uint32_t condition) {
  division_vector_run<Width, Ops>(mask, mode, d, a, b, c, condition);
}

template void div_fmas_x86_64_v3<32>(uint32_t, uint32_t, uint32_t *const *, const uint32_t *const *,
                                     const uint32_t *const *, const uint32_t *const *, uint32_t);
template void div_fmas_x86_64_v3<64>(uint32_t, uint32_t, uint32_t *const *, const uint32_t *const *,
                                     const uint32_t *const *, const uint32_t *const *, uint32_t);

} // namespace goc

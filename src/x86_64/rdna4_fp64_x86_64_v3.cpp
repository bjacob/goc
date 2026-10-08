// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_fp64.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {
namespace {

__m256d load(const uint32_t *const *v, int lane, __m256i keep, __m256i flip) {
  auto low = _mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i *>(v[0] + lane)));
  auto high =
      _mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i *>(v[1] + lane)));
  auto bits = _mm256_or_si256(low, _mm256_slli_epi64(high, 32));
  return _mm256_castsi256_pd(_mm256_xor_si256(_mm256_and_si256(bits, keep), flip));
}

template <Fp64 Op>
void run(uint32_t mask, uint32_t mode, uint32_t *const *d, const uint32_t *const *a,
         const uint32_t *const *b, const uint32_t *const *c) {
  auto ka = _mm256_set1_epi64x(mode & GOC_ALU_ABS_A ? INT64_MAX : -1);
  auto kb = _mm256_set1_epi64x(mode & GOC_ALU_ABS_B ? INT64_MAX : -1);
  auto kc = _mm256_set1_epi64x(mode & GOC_ALU_ABS_C ? INT64_MAX : -1);
  auto na = _mm256_set1_epi64x(mode & GOC_ALU_NEG_A ? INT64_MIN : 0);
  auto nb = _mm256_set1_epi64x(mode & GOC_ALU_NEG_B ? INT64_MIN : 0);
  auto nc = _mm256_set1_epi64x(mode & GOC_ALU_NEG_C ? INT64_MIN : 0);
  const double scales[] = {1, 2, 4, 0.5};
  auto scale = _mm256_set1_pd(scales[(mode >> 6) & 3]);
  uint32_t result[2][32];
  for (int lane = 0; lane < 32; lane += 4) {
    auto x = load(a, lane, ka, na), y = _mm256_setzero_pd();
    if constexpr (fp64_sources(Op) >= 2)
      y = load(b, lane, kb, nb);
    __m256d value;
    if constexpr (Op == Fp64::Add)
      value = _mm256_add_pd(x, y);
    if constexpr (Op == Fp64::Mul)
      value = _mm256_mul_pd(x, y);
    if constexpr (Op == Fp64::Fma)
      value = _mm256_fmadd_pd(x, y, load(c, lane, kc, nc));
    if constexpr (Op == Fp64::Trunc)
      value = _mm256_round_pd(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    if constexpr (Op == Fp64::Ceil)
      value = _mm256_round_pd(x, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
    if constexpr (Op == Fp64::Rndne)
      value = _mm256_round_pd(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    if constexpr (Op == Fp64::Floor)
      value = _mm256_round_pd(x, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    if constexpr (Op == Fp64::Sqrt || Op == Fp64::Rsq)
      value = _mm256_sqrt_pd(x);
    if constexpr (Op == Fp64::Rcp)
      value = _mm256_div_pd(_mm256_set1_pd(1), x);
    if constexpr (Op == Fp64::Rsq)
      value = _mm256_div_pd(_mm256_set1_pd(1), value);
    if constexpr (Op == Fp64::Fract) {
      value = _mm256_sub_pd(x, _mm256_floor_pd(x));
      auto limit = _mm256_castsi256_pd(_mm256_set1_epi64x(INT64_C(0x3fefffffffffffff)));
      value = _mm256_blendv_pd(value, limit, _mm256_cmp_pd(value, limit, _CMP_GT_OQ));
    }
    if (mode & GOC_ALU_OMOD_HALF)
      value = _mm256_mul_pd(value, scale);
    if (mode & GOC_ALU_CLAMP)
      value = _mm256_min_pd(_mm256_max_pd(value, _mm256_setzero_pd()), _mm256_set1_pd(1));
    // Gather low words and high words into separate 128-bit halves.
    auto words = _mm256_permutevar8x32_epi32(_mm256_castpd_si256(value),
                                             _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(result[0] + lane), _mm256_castsi256_si128(words));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(result[1] + lane),
                     _mm256_extracti128_si256(words, 1));
  }
  // Source halves can alias either output half, so delay all masked writes.
  for (int reg = 0; reg < 2; ++reg)
    for (int lane = 0; lane < 32; lane += 8) {
      auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(mask >> lane)),
                                      _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
      _mm256_maskstore_epi32(
          reinterpret_cast<int *>(d[reg] + lane), active,
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(result[reg] + lane)));
    }
}

} // namespace

void fp64_x86_64_v3(Fp64 op, uint32_t mask, uint32_t mode, uint32_t *const *d,
                    const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  switch (op) {
  case Fp64::Trunc:
    return run<Fp64::Trunc>(mask, mode, d, a, b, c);
  case Fp64::Ceil:
    return run<Fp64::Ceil>(mask, mode, d, a, b, c);
  case Fp64::Rndne:
    return run<Fp64::Rndne>(mask, mode, d, a, b, c);
  case Fp64::Floor:
    return run<Fp64::Floor>(mask, mode, d, a, b, c);
  case Fp64::Fract:
    return run<Fp64::Fract>(mask, mode, d, a, b, c);
  case Fp64::Sqrt:
    return run<Fp64::Sqrt>(mask, mode, d, a, b, c);
  case Fp64::Rcp:
    return run<Fp64::Rcp>(mask, mode, d, a, b, c);
  case Fp64::Rsq:
    return run<Fp64::Rsq>(mask, mode, d, a, b, c);
  case Fp64::Add:
    return run<Fp64::Add>(mask, mode, d, a, b, c);
  case Fp64::Mul:
    return run<Fp64::Mul>(mask, mode, d, a, b, c);
  case Fp64::Fma:
    return run<Fp64::Fma>(mask, mode, d, a, b, c);
  }
}

} // namespace goc

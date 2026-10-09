// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "ldexp.h"
#include "x86_64/alu_x86_64_v4.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {
namespace {

template <bool Fp64>
void run(uint32_t exec_mask, uint32_t mode, uint32_t *const *d, const uint32_t *const *a,
         const uint32_t *b) {
  uint32_t result[Fp64 ? 2 : 1][32];
  const double scales[] = {1, 2, 4, 0.5};
  for (int lane = 0; lane < 32; lane += Fp64 ? 8 : 16) {
    if constexpr (Fp64) {
      auto low =
          _mm512_cvtepu32_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a[0] + lane)));
      auto high =
          _mm512_cvtepu32_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a[1] + lane)));
      auto bits = _mm512_or_si512(low, _mm512_slli_epi64(high, 32));
      if (mode & GOC_ALU_ABS_A)
        bits = _mm512_and_si512(bits, _mm512_set1_epi64(INT64_MAX));
      if (mode & GOC_ALU_NEG_A)
        bits = _mm512_xor_si512(bits, _mm512_set1_epi64(INT64_MIN));
      auto exponent =
          _mm512_cvtepi32_pd(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane)));
      auto value = _mm512_scalef_pd(_mm512_castsi512_pd(bits), exponent);
      if (mode & GOC_ALU_OMOD_HALF) {
        auto exact_exponent = _mm512_add_pd(_mm512_getexp_pd(_mm512_castsi512_pd(bits)), exponent);
        auto tiny_before_rounding =
            _mm512_cmp_pd_mask(exact_exponent, _mm512_set1_pd(-1022), _CMP_LT_OQ);
        value = _mm512_mask_mov_pd(value, tiny_before_rounding, _mm512_setzero_pd());
        value = prepare_omod_f64(value, mode);
        value = _mm512_mul_pd(value, _mm512_set1_pd(scales[(mode >> 6) & 3]));
      }
      if (mode & GOC_ALU_CLAMP)
        value = _mm512_min_pd(_mm512_max_pd(value, _mm512_setzero_pd()), _mm512_set1_pd(1));
      bits = _mm512_castpd_si512(value);
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[0] + lane),
                          _mm512_cvtepi64_epi32(bits));
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[1] + lane),
                          _mm512_cvtepi64_epi32(_mm512_srli_epi64(bits, 32)));
    } else {
      auto bits = _mm512_loadu_si512(a[0] + lane);
      if (mode & GOC_ALU_ABS_A)
        bits = _mm512_and_si512(bits, _mm512_set1_epi32(INT32_MAX));
      if (mode & GOC_ALU_NEG_A)
        bits = _mm512_xor_si512(bits, _mm512_set1_epi32(INT32_MIN));
      auto exponent = _mm512_loadu_si512(b + lane);
      // Exponents beyond these bounds have the same overflow/underflow result;
      // bounding them also makes the integer-to-float conversion exact.
      exponent = _mm512_min_epi32(_mm512_max_epi32(exponent, _mm512_set1_epi32(-4096)),
                                  _mm512_set1_epi32(4096));
      auto value = _mm512_scalef_ps(_mm512_castsi512_ps(bits), _mm512_cvtepi32_ps(exponent));
      if (mode & GOC_ALU_OMOD_HALF) {
        auto exact_exponent = _mm512_add_ps(_mm512_getexp_ps(_mm512_castsi512_ps(bits)),
                                            _mm512_cvtepi32_ps(exponent));
        auto tiny_before_rounding =
            _mm512_cmp_ps_mask(exact_exponent, _mm512_set1_ps(-126), _CMP_LT_OQ);
        value = _mm512_mask_mov_ps(value, tiny_before_rounding, _mm512_setzero_ps());
        value = prepare_omod_f32(value, mode);
        value = _mm512_mul_ps(value, _mm512_set1_ps(scales[(mode >> 6) & 3]));
      }
      if (mode & GOC_ALU_CLAMP)
        value = _mm512_min_ps(_mm512_max_ps(value, _mm512_setzero_ps()), _mm512_set1_ps(1));
      _mm512_storeu_si512(result[0] + lane, _mm512_castps_si512(value));
    }
  }
  for (int reg = 0; reg < (Fp64 ? 2 : 1); ++reg)
    for (int lane = 0; lane < 32; lane += 16)
      _mm512_mask_storeu_epi32(d[reg] + lane, __mmask16(exec_mask >> lane),
                               _mm512_loadu_si512(result[reg] + lane));
}

} // namespace

void ldexp_x86_64_v4(bool fp64, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *b) {
  if (fp64)
    run<true>(exec_mask, mode, d, a, b);
  else
    run<false>(exec_mask, mode, d, a, b);
}

} // namespace goc

// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_trig.h"
#include "rdna4_trig_coefficients.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {
namespace {

template <bool Cosine> __m256 evaluate(__m256i bits) {
  const __m256i zero = _mm256_setzero_si256();
  const __m256i one = _mm256_set1_epi32(1);
  const __m256i magnitude = _mm256_and_si256(bits, _mm256_set1_epi32(0x7fffffff));
  const __m256i mantissa = _mm256_or_si256(_mm256_and_si256(magnitude, _mm256_set1_epi32(0x7fffff)),
                                           _mm256_set1_epi32(0x800000));
  // Reduce in turns. Variable shifts with counts >=32 yield zero, including
  // negative counts represented as unsigned, so neither large nor tiny inputs
  // require a float-to-integer conversion or a separate reduction path.
  const __m256i shift = _mm256_sub_epi32(_mm256_srli_epi32(magnitude, 23), _mm256_set1_epi32(119));
  const __m256i phase =
      _mm256_and_si256(_mm256_or_si256(_mm256_sllv_epi32(mantissa, shift),
                                       _mm256_srlv_epi32(mantissa, _mm256_sub_epi32(zero, shift))),
                       _mm256_set1_epi32(0x7fffffff));
  const __m256i quadrant = _mm256_srli_epi32(phase, 29);
  const __m256i quarter = _mm256_and_si256(phase, _mm256_set1_epi32(0x1fffffff));
  const __m256i reflected = _mm256_cmpgt_epi32(quarter, _mm256_set1_epi32(0x0fffffff));
  const __m256i reduced = _mm256_blendv_epi8(
      quarter, _mm256_sub_epi32(_mm256_set1_epi32(0x20000000), quarter), reflected);
  const __m256i table = _mm256_and_si256(
      _mm256_xor_si256(_mm256_xor_si256(quadrant, reflected), _mm256_set1_epi32(Cosine)), one);
  const __m256i index = _mm256_srli_epi32(_mm256_add_epi32(reduced, reflected), 24);
  const __m256i fraction = _mm256_sub_epi32(reduced, _mm256_slli_epi32(index, 24));
  const __m256i column = _mm256_add_epi32(index, _mm256_slli_epi32(table, 4));
  const __m256 t = _mm256_mul_ps(_mm256_cvtepi32_ps(fraction), _mm256_set1_ps(0x1p-24f));
  const __m256 c0 = _mm256_cvtepi32_ps(_mm256_i32gather_epi32(trig::coefficients[0], column, 4));
  const __m256 c1 = _mm256_cvtepi32_ps(_mm256_i32gather_epi32(trig::coefficients[1], column, 4));
  const __m256 c2 = _mm256_cvtepi32_ps(_mm256_i32gather_epi32(trig::coefficients[2], column, 4));
  const __m256 c3 = _mm256_cvtepi32_ps(_mm256_i32gather_epi32(trig::coefficients[3], column, 4));
  __m256 value =
      _mm256_mul_ps(_mm256_fmadd_ps(t, _mm256_fmadd_ps(t, _mm256_fmadd_ps(t, c3, c2), c1), c0),
                    _mm256_set1_ps(0x1p-26f));
  __m256i sign = _mm256_srli_epi32(quadrant, 1);
  if constexpr (Cosine)
    sign = _mm256_xor_si256(sign, quadrant);
  else
    sign = _mm256_xor_si256(sign, _mm256_srli_epi32(bits, 31));
  // Captured phase zeros are +0; the original small-sine region preserves -0.
  sign = _mm256_andnot_si256(
      _mm256_castps_si256(_mm256_cmp_ps(value, _mm256_setzero_ps(), _CMP_EQ_OQ)),
      _mm256_slli_epi32(sign, 31));
  value = _mm256_xor_ps(value, _mm256_castsi256_ps(sign));
  if constexpr (!Cosine) {
    const __m256i tiny = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x39c00000), magnitude);
    // Limit this multiplication to tiny inputs to avoid spurious overflow for
    // huge finite values and invalid exceptions for nonfinite unused lanes.
    __m256 small = _mm256_castsi256_ps(_mm256_and_si256(bits, tiny));
    small = _mm256_mul_ps(small, _mm256_castsi256_ps(_mm256_set1_epi32(0x40c90fd5)));
    value = _mm256_blendv_ps(value, small, _mm256_castsi256_ps(tiny));
  }
  const __m256i nonfinite = _mm256_cmpgt_epi32(magnitude, _mm256_set1_epi32(0x7f7fffff));
  const __m256i infinity = _mm256_cmpeq_epi32(magnitude, _mm256_set1_epi32(0x7f800000));
  const __m256i nan = _mm256_blendv_epi8(_mm256_or_si256(bits, _mm256_set1_epi32(0x400000)),
                                         _mm256_set1_epi32(int32_t(0xffc00000)), infinity);
  return _mm256_blendv_ps(value, _mm256_castsi256_ps(nan), _mm256_castsi256_ps(nonfinite));
}

template <bool Cosine> void run(uint32_t mask, uint32_t mode, uint32_t *d, const uint32_t *a) {
  const __m256i keep = _mm256_set1_epi32(mode & GOC_ALU_ABS_A ? 0x7fffffff : -1);
  const __m256i flip = _mm256_set1_epi32(mode & GOC_ALU_NEG_A ? INT32_MIN : 0);
  const float scales[] = {1, 2, 4, 0.5f};
  const __m256 scale = _mm256_set1_ps(scales[(mode >> 6) & 3]);
  for (int lane = 0; lane < 32; lane += 8) {
    const __m256i bits = _mm256_xor_si256(
        _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane)), keep),
        flip);
    __m256 value = evaluate<Cosine>(bits);
    if (mode & GOC_ALU_OMOD_HALF)
      value = _mm256_mul_ps(value, scale);
    if (mode & GOC_ALU_CLAMP)
      value = _mm256_min_ps(_mm256_max_ps(value, _mm256_setzero_ps()), _mm256_set1_ps(1));
    if (mode & GOC_ALU_OMOD_HALF) {
      const __m256i magnitude =
          _mm256_and_si256(_mm256_castps_si256(value), _mm256_set1_epi32(0x7fffffff));
      value = _mm256_andnot_ps(
          _mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_set1_epi32(0x800000), magnitude)), value);
    }
    const __m256i active = _mm256_sllv_epi32(_mm256_set1_epi32(int(mask >> lane)),
                                             _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), active, _mm256_castps_si256(value));
  }
}

} // namespace

void trig_x86_64_v3(bool cosine, uint32_t mask, uint32_t mode, uint32_t *d, const uint32_t *a) {
  if (cosine)
    run<true>(mask, mode, d, a);
  else
    run<false>(mask, mode, d, a);
}

} // namespace goc

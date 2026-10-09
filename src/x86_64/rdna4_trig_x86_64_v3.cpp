// SPDX-License-Identifier: MIT

#include "x86_64/rdna4_trig_x86_64_v3.h"
#include "goc/goc.h"
#include "rdna4_trig.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {
namespace {

template <bool Cosine> void run(uint32_t mask, uint32_t mode, uint32_t *d, const uint32_t *a) {
  const __m256i keep = _mm256_set1_epi32(mode & GOC_ALU_ABS_A ? 0x7fffffff : -1);
  const __m256i flip = _mm256_set1_epi32(mode & GOC_ALU_NEG_A ? INT32_MIN : 0);
  const float scales[] = {1, 2, 4, 0.5f};
  const __m256 scale = _mm256_set1_ps(scales[(mode >> 6) & 3]);
  for (int lane = 0; lane < 32; lane += 8) {
    const __m256i bits = _mm256_xor_si256(
        _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane)), keep),
        flip);
    __m256 value = trig_value<Cosine>(bits);
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

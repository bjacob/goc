// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_byte_conversion.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {

template <unsigned Byte>
void byte_conversion_x86_64_v3(uint32_t mask, uint32_t mode, uint32_t *d, const uint32_t *a) {
  const float scales[] = {1, 2, 4, 0.5f};
  auto scale = _mm256_set1_ps(scales[(mode >> 6) & 3]);
  for (unsigned lane = 0; lane < 32; lane += 8) {
    auto raw = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    raw = _mm256_and_si256(_mm256_srli_epi32(raw, 8 * Byte), _mm256_set1_epi32(255));
    auto value = _mm256_mul_ps(_mm256_cvtepi32_ps(raw), scale);
    if (mode & GOC_ALU_CLAMP)
      value = _mm256_min_ps(value, _mm256_set1_ps(1));
    auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(mask >> lane)),
                                    _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), active, _mm256_castps_si256(value));
  }
}

template void byte_conversion_x86_64_v3<0>(uint32_t, uint32_t, uint32_t *, const uint32_t *);
template void byte_conversion_x86_64_v3<1>(uint32_t, uint32_t, uint32_t *, const uint32_t *);
template void byte_conversion_x86_64_v3<2>(uint32_t, uint32_t, uint32_t *, const uint32_t *);
template void byte_conversion_x86_64_v3<3>(uint32_t, uint32_t, uint32_t *, const uint32_t *);

} // namespace goc

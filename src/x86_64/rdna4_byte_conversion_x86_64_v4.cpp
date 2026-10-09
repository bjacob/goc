// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_byte_conversion.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {

template <unsigned Byte>
void byte_conversion_x86_64_v4(uint32_t mask, uint32_t mode, uint32_t *d, const uint32_t *a) {
  const float scales[] = {1, 2, 4, 0.5f};
  auto scale = _mm512_set1_ps(scales[(mode >> 6) & 3]);
  for (unsigned lane = 0; lane < 32; lane += 16) {
    auto raw = _mm512_loadu_si512(a + lane);
    raw = _mm512_and_si512(_mm512_srli_epi32(raw, 8 * Byte), _mm512_set1_epi32(255));
    auto value = _mm512_mul_ps(_mm512_cvtepi32_ps(raw), scale);
    if (mode & GOC_ALU_CLAMP)
      value = _mm512_min_ps(value, _mm512_set1_ps(1));
    _mm512_mask_storeu_epi32(d + lane, __mmask16(mask >> lane), _mm512_castps_si512(value));
  }
}

template void byte_conversion_x86_64_v4<0>(uint32_t, uint32_t, uint32_t *, const uint32_t *);
template void byte_conversion_x86_64_v4<1>(uint32_t, uint32_t, uint32_t *, const uint32_t *);
template void byte_conversion_x86_64_v4<2>(uint32_t, uint32_t, uint32_t *, const uint32_t *);
template void byte_conversion_x86_64_v4<3>(uint32_t, uint32_t, uint32_t *, const uint32_t *);

} // namespace goc

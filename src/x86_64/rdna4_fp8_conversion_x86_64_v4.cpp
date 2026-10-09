// SPDX-License-Identifier: MIT

#include "rdna4_fp8_conversion.h"
#include "x86_64/rdna4_fp8_x86_64_v4.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {

template <bool Bf8, bool Packed>
void fp8_conversion_x86_64_v4(uint32_t mask, unsigned shift, uint32_t *const *d,
                              const uint32_t *a) {
  for (unsigned lane = 0; lane < 32; lane += 16) {
    auto raw = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + lane));
    raw = _mm512_srl_epi32(raw, _mm_cvtsi32_si128(int(shift)));
    auto low = _mm512_castps_si512(widen_fp8<Bf8>(raw));
    auto active = __mmask16(mask >> lane);
    if constexpr (Packed) {
      auto high = _mm512_castps_si512(widen_fp8<Bf8>(_mm512_srli_epi32(raw, 8)));
      _mm512_mask_storeu_epi32(d[0] + lane, active, low);
      _mm512_mask_storeu_epi32(d[1] + lane, active, high);
    } else {
      _mm512_mask_storeu_epi32(d[0] + lane, active, low);
    }
  }
}

template void fp8_conversion_x86_64_v4<false, false>(uint32_t, unsigned, uint32_t *const *,
                                                     const uint32_t *);
template void fp8_conversion_x86_64_v4<true, false>(uint32_t, unsigned, uint32_t *const *,
                                                    const uint32_t *);
template void fp8_conversion_x86_64_v4<false, true>(uint32_t, unsigned, uint32_t *const *,
                                                    const uint32_t *);
template void fp8_conversion_x86_64_v4<true, true>(uint32_t, unsigned, uint32_t *const *,
                                                   const uint32_t *);

} // namespace goc

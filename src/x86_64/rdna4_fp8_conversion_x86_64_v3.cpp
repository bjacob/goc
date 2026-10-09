// SPDX-License-Identifier: MIT

#include "rdna4_fp8_conversion.h"
#include "x86_64/rdna4_fp8.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {

template <bool Bf8, bool Packed>
void fp8_conversion_x86_64_v3(uint32_t mask, unsigned shift, uint32_t *const *d,
                              const uint32_t *a) {
  for (unsigned lane = 0; lane < 32; lane += 8) {
    auto raw = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    raw = _mm256_srl_epi32(raw, _mm_cvtsi32_si128(int(shift)));
    auto low = _mm256_castps_si256(widen_fp8<Bf8>(raw));
    auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(mask >> lane)),
                                    _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    if constexpr (Packed) {
      auto high = _mm256_castps_si256(widen_fp8<Bf8>(_mm256_srli_epi32(raw, 8)));
      _mm256_maskstore_epi32(reinterpret_cast<int *>(d[0] + lane), active, low);
      _mm256_maskstore_epi32(reinterpret_cast<int *>(d[1] + lane), active, high);
    } else {
      _mm256_maskstore_epi32(reinterpret_cast<int *>(d[0] + lane), active, low);
    }
  }
}

template void fp8_conversion_x86_64_v3<false, false>(uint32_t, unsigned, uint32_t *const *,
                                                     const uint32_t *);
template void fp8_conversion_x86_64_v3<true, false>(uint32_t, unsigned, uint32_t *const *,
                                                    const uint32_t *);
template void fp8_conversion_x86_64_v3<false, true>(uint32_t, unsigned, uint32_t *const *,
                                                    const uint32_t *);
template void fp8_conversion_x86_64_v3<true, true>(uint32_t, unsigned, uint32_t *const *,
                                                   const uint32_t *);

} // namespace goc

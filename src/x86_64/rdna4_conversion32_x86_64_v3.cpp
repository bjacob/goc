// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_conversion32.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {

template <Conversion32 Op>
void conversion32_x86_64_v3(uint32_t mask, uint32_t mode, uint32_t *d, const uint32_t *a) {
  constexpr bool to_float =
      Op == Conversion32::SignedToFloat || Op == Conversion32::UnsignedToFloat;
  const auto zero = _mm256_setzero_ps();
  const float scales[] = {1, 2, 4, 0.5f};
  const auto scale = _mm256_set1_ps(scales[(mode >> 6) & 3]);
  const auto abs_mask = _mm256_set1_epi32(mode & GOC_ALU_ABS_A ? 0x7fffffff : -1);
  const auto neg_mask = _mm256_set1_epi32(mode & GOC_ALU_NEG_A ? int32_t(INT32_MIN) : 0);
  for (int lane = 0; lane < 32; lane += 8) {
    auto raw = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    __m256i result;
    if constexpr (to_float) {
      __m256 x;
      if constexpr (Op == Conversion32::SignedToFloat)
        x = _mm256_cvtepi32_ps(raw);
      else {
        // Both 16-bit pieces convert exactly; the FMA rounds the unsigned
        // value once, avoiding signed-conversion/2^32 double rounding.
        auto high = _mm256_cvtepi32_ps(_mm256_srli_epi32(raw, 16));
        auto low = _mm256_cvtepi32_ps(_mm256_and_si256(raw, _mm256_set1_epi32(65535)));
        x = _mm256_fmadd_ps(high, _mm256_set1_ps(65536), low);
      }
      x = _mm256_mul_ps(x, scale);
      if (mode & GOC_ALU_CLAMP)
        x = _mm256_min_ps(_mm256_max_ps(x, zero), _mm256_set1_ps(1));
      result = _mm256_castps_si256(x);
    } else {
      raw = _mm256_xor_si256(_mm256_and_si256(raw, abs_mask), neg_mask);
      auto x = _mm256_castsi256_ps(raw);
      {
        auto nan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
        auto overflow = _mm256_cmp_ps(x, _mm256_set1_ps(2147483648.0f), _CMP_GE_OQ);
        x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-2147483648.0f)),
                          _mm256_set1_ps(2147483520.0f));
        if constexpr (Op == Conversion32::Nearest) {
          auto lower = _mm256_floor_ps(x);
          auto up = _mm256_cmp_ps(_mm256_sub_ps(x, lower), _mm256_set1_ps(0.5f), _CMP_GE_OQ);
          x = _mm256_add_ps(lower, _mm256_and_ps(up, _mm256_set1_ps(1)));
        }
        if constexpr (Op == Conversion32::Floor)
          x = _mm256_floor_ps(x);
        result = _mm256_cvttps_epi32(x);
        result =
            _mm256_blendv_epi8(result, _mm256_set1_epi32(INT32_MAX), _mm256_castps_si256(overflow));
        auto nan_result =
            _mm256_xor_si256(_mm256_srai_epi32(raw, 31), _mm256_set1_epi32(INT32_MAX));
        result = _mm256_blendv_epi8(result, nan_result, _mm256_castps_si256(nan));
      }
    }
    auto active = _mm256_set1_epi32(int(mask >> lane));
    active = _mm256_sllv_epi32(active, _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), active, result);
  }
}

template void conversion32_x86_64_v3<Conversion32::SignedToFloat>(uint32_t, uint32_t, uint32_t *,
                                                                  const uint32_t *);
template void conversion32_x86_64_v3<Conversion32::UnsignedToFloat>(uint32_t, uint32_t, uint32_t *,
                                                                    const uint32_t *);
template void conversion32_x86_64_v3<Conversion32::Nearest>(uint32_t, uint32_t, uint32_t *,
                                                            const uint32_t *);
template void conversion32_x86_64_v3<Conversion32::Floor>(uint32_t, uint32_t, uint32_t *,
                                                          const uint32_t *);

} // namespace goc

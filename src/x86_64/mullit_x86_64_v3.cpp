// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "mullit.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {

void mullit_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                      const uint32_t *b, const uint32_t *c) {
  auto bits = [](uint32_t x) { return _mm256_castsi256_ps(_mm256_set1_epi32(int(x))); };
  auto zero = _mm256_setzero_ps();
  auto sentinel = bits(0xff7fffff);
  auto exponent = bits(0x7f800000);
  unsigned scale = (mode >> 6) & 3;
  for (unsigned lane = 0; lane < 32; lane += 8) {
    auto input = [&](const uint32_t *source, unsigned modifiers) {
      auto value =
          _mm256_castsi256_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(source + lane)));
      value = _mm256_and_ps(value, bits(modifiers & GOC_ALU_ABS_A ? 0x7fffffff : UINT32_MAX));
      return _mm256_xor_ps(value, bits(modifiers & GOC_ALU_NEG_A ? 0x80000000 : 0));
    };
    auto x = input(a, mode), y = input(b, mode >> 1), z = input(c, mode >> 2);
    auto bad_b = _mm256_or_ps(_mm256_cmp_ps(y, sentinel, _CMP_EQ_OQ),
                              _mm256_cmp_ps(y, bits(0xff800000), _CMP_EQ_OQ));
    auto invalid = _mm256_or_ps(bad_b, _mm256_or_ps(_mm256_cmp_ps(y, y, _CMP_UNORD_Q),
                                                    _mm256_cmp_ps(z, zero, _CMP_NGT_UQ)));
    auto factor_zero =
        _mm256_or_ps(_mm256_cmp_ps(x, zero, _CMP_EQ_OQ), _mm256_cmp_ps(y, zero, _CMP_EQ_OQ));
    auto value = _mm256_mul_ps(x, y);
    value = _mm256_blendv_ps(value, zero, factor_zero);
    value = _mm256_blendv_ps(value, sentinel, invalid);
    if (scale) {
      auto tiny = _mm256_cmp_ps(_mm256_and_ps(value, exponent), zero, _CMP_EQ_OQ);
      value = _mm256_blendv_ps(value, zero, tiny);
      value = _mm256_mul_ps(value, _mm256_set1_ps(scale == 1 ? 2.0f : scale == 2 ? 4.0f : 0.5f));
      tiny = _mm256_cmp_ps(_mm256_and_ps(value, exponent), zero, _CMP_EQ_OQ);
      value = _mm256_blendv_ps(value, _mm256_and_ps(value, bits(0x80000000)), tiny);
    }
    if (mode & GOC_ALU_CLAMP)
      value = _mm256_min_ps(_mm256_max_ps(value, zero), _mm256_set1_ps(1.0f));
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                            _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask,
                           _mm256_castps_si256(value));
  }
}

} // namespace goc

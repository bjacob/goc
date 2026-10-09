// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_mullit.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {

void mullit_x86_64_v4(uint32_t mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                      const uint32_t *b, const uint32_t *c) {
  auto bits = [](uint32_t x) { return _mm512_castsi512_ps(_mm512_set1_epi32(int(x))); };
  auto zero = _mm512_setzero_ps();
  auto sentinel = bits(0xff7fffff);
  auto exponent = bits(0x7f800000);
  unsigned scale = (mode >> 6) & 3;
  for (unsigned lane = 0; lane < 32; lane += 16) {
    auto input = [&](const uint32_t *source, unsigned modifiers) {
      auto value =
          _mm512_castsi512_ps(_mm512_loadu_si512(reinterpret_cast<const __m512i *>(source + lane)));
      value = _mm512_and_ps(value, bits(modifiers & GOC_ALU_ABS_A ? 0x7fffffff : UINT32_MAX));
      return _mm512_xor_ps(value, bits(modifiers & GOC_ALU_NEG_A ? 0x80000000 : 0));
    };
    auto x = input(a, mode), y = input(b, mode >> 1), z = input(c, mode >> 2);
    auto bad_b = (_mm512_cmp_ps_mask(y, sentinel, _CMP_EQ_OQ) |
                  _mm512_cmp_ps_mask(y, bits(0xff800000), _CMP_EQ_OQ));
    auto invalid = (bad_b | (_mm512_cmp_ps_mask(y, y, _CMP_UNORD_Q) |
                             _mm512_cmp_ps_mask(z, zero, _CMP_NGT_UQ)));
    auto factor_zero =
        (_mm512_cmp_ps_mask(x, zero, _CMP_EQ_OQ) | _mm512_cmp_ps_mask(y, zero, _CMP_EQ_OQ));
    auto value = _mm512_mul_ps(x, y);
    value = _mm512_mask_blend_ps(factor_zero, value, zero);
    value = _mm512_mask_blend_ps(invalid, value, sentinel);
    if (scale) {
      auto tiny = _mm512_cmp_ps_mask(_mm512_and_ps(value, exponent), zero, _CMP_EQ_OQ);
      value = _mm512_mask_blend_ps(tiny, value, zero);
      value = _mm512_mul_ps(value, _mm512_set1_ps(scale == 1 ? 2.0f : scale == 2 ? 4.0f : 0.5f));
      tiny = _mm512_cmp_ps_mask(_mm512_and_ps(value, exponent), zero, _CMP_EQ_OQ);
      value = _mm512_mask_blend_ps(tiny, value, _mm512_and_ps(value, bits(0x80000000)));
    }
    if (mode & GOC_ALU_CLAMP)
      value = _mm512_min_ps(_mm512_max_ps(value, zero), _mm512_set1_ps(1.0f));
    _mm512_mask_storeu_epi32(d + lane, __mmask16(mask >> lane), _mm512_castps_si512(value));
  }
}

} // namespace goc

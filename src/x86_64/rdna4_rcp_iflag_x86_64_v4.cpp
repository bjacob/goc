// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_rcp_iflag.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {

uint32_t rcp_iflag_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a) {
  auto exponent = _mm512_set1_epi32(0x7f800000), sign = _mm512_set1_epi32(int(0x80000000u)),
       zero = _mm512_setzero_si512();
  auto keep = _mm512_set1_epi32(mode & GOC_ALU_ABS_A ? 0x7fffffff : -1),
       flip = _mm512_set1_epi32(mode & GOC_ALU_NEG_A ? int(0x80000000u) : 0);
  const float scales[] = {1, 2, 4, 0.5f};
  auto scale = _mm512_set1_ps(scales[(mode >> 6) & 3]);
  uint32_t zeros = 0;
  for (unsigned lane = 0; lane < 32; lane += 16) {
    auto raw = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + lane));
    auto tiny = _mm512_cmpeq_epi32_mask(_mm512_and_si512(raw, exponent), zero);
    zeros |= uint32_t(tiny) << lane;
    raw = _mm512_xor_si512(_mm512_and_si512(raw, keep), flip);
    raw = _mm512_mask_blend_epi32(tiny, raw, _mm512_and_si512(raw, sign));
    auto value = _mm512_div_ps(_mm512_set1_ps(1), _mm512_castsi512_ps(raw));
    auto bits = _mm512_castps_si512(value);
    bits = _mm512_mask_blend_epi32(_mm512_cmpeq_epi32_mask(_mm512_and_si512(bits, exponent), zero),
                                   bits, _mm512_and_si512(bits, sign));
    if (mode & GOC_ALU_OMOD_HALF) {
      bits = _mm512_mask_blend_epi32(
          _mm512_cmpeq_epi32_mask(_mm512_and_si512(bits, _mm512_set1_epi32(0x7fffffff)), zero),
          bits, zero);
      value = _mm512_mul_ps(_mm512_castsi512_ps(bits), scale);
    } else
      value = _mm512_castsi512_ps(bits);
    if (mode & GOC_ALU_CLAMP)
      value = _mm512_min_ps(_mm512_max_ps(value, _mm512_setzero_ps()), _mm512_set1_ps(1));
    bits = _mm512_castps_si512(value);
    bits = _mm512_mask_blend_epi32(_mm512_cmpeq_epi32_mask(_mm512_and_si512(bits, exponent), zero),
                                   bits, _mm512_and_si512(bits, sign));
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), bits);
  }
  return (zeros & exec_mask) && !(mode & GOC_ALU_CLAMP) ? GOC_RDNA4_EXCEPTION_INT_DIV0 : 0;
}

} // namespace goc

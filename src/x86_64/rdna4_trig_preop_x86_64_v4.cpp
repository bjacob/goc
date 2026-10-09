// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_trig_preop.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {

void trig_preop_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                          const uint32_t *a_hi, const uint32_t *b) {
  uint32_t result[2][32];
  for (unsigned lane = 0; lane < 32; lane += 8) {
    auto a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a_hi + lane));
    auto selector = _mm256_and_si256(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane)), _mm256_set1_epi32(31));
    auto exponent = _mm256_and_si256(_mm256_srli_epi32(a, 20), _mm256_set1_epi32(2047));
    auto shift =
        _mm256_add_epi32(_mm256_mullo_epi32(selector, _mm256_set1_epi32(53)),
                         _mm256_max_epi32(_mm256_sub_epi32(exponent, _mm256_set1_epi32(1077)),
                                          _mm256_setzero_si256()));
    auto bank = _mm256_and_si256(_mm256_cmpgt_epi32(exponent, _mm256_set1_epi32(1967)),
                                 _mm256_set1_epi32(1185));
    auto index = _mm256_add_epi32(bank, _mm256_min_epi32(shift, _mm256_set1_epi32(1184)));
    auto value = _mm512_i32gather_epi64(index, trig_preop_table.data(), 8);
    unsigned omod = (mode >> 6) & 3;
    if (omod) {
      auto normal = _mm512_cmpgt_epi64_mask(value, _mm512_set1_epi64(0xfffffffffffff));
      value = _mm512_add_epi64(
          value, _mm512_set1_epi64(int64_t(omod == 3 ? -1 : int(omod)) * (INT64_C(1) << 52)));
      normal &= _mm512_cmpgt_epi64_mask(value, _mm512_set1_epi64(0xfffffffffffff));
      value = _mm512_maskz_mov_epi64(normal, value);
    }
    if (mode & GOC_ALU_CLAMP) {
      value = _mm512_min_epi64(value, _mm512_set1_epi64(0x3ff0000000000000));
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[0] + lane),
                        _mm512_cvtepi64_epi32(value));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[1] + lane),
                        _mm512_cvtepi64_epi32(_mm512_srli_epi64(value, 32)));
  }
  for (unsigned reg = 0; reg < 2; ++reg)
    for (unsigned lane = 0; lane < 32; lane += 16)
      _mm512_mask_storeu_epi32(d[reg] + lane, __mmask16(exec_mask >> lane),
                               _mm512_loadu_si512(result[reg] + lane));
}

} // namespace goc

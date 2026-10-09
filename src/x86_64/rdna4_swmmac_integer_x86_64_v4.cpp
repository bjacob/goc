// SPDX-License-Identifier: MIT

#include "rdna4_swmmac_integer.h"
#include "x86_64/rdna4_alu_x86_64_v4.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {

void swmmac_integer_x86_64_v4(unsigned k, bool clamp, uint32_t *const *d,
                              const SwmmacIntegerInputs &input) {
  uint32_t result[8][32];
  for (unsigned row = 0; row < 16; ++row)
    for (unsigned col = 0; col < 16; col += 16) {
      auto acc = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(input.acc[row] + col));
      for (unsigned stage = 0; stage < 2; ++stage) {
        auto sum = _mm512_setzero_si512();
        for (unsigned local = 0; local < k / 4; ++local) {
          unsigned ck = swmmac_integer_ck(stage, local);
          auto b = _mm512_loadu_si512(
              reinterpret_cast<const __m512i *>(input.b[input.selected[row][ck]] + col));
          sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(_mm512_set1_epi32(input.a[row][ck]), b));
        }
        auto next = _mm512_add_epi32(acc, sum);
        if (clamp)
          next = saturate_signed_sum(acc, sum, next);
        acc = next;
      }
      _mm512_storeu_si512(reinterpret_cast<__m512i *>(result[row % 8] + col + 16 * (row / 8)), acc);
    }
  for (unsigned reg = 0; reg < 8; ++reg)
    for (unsigned lane = 0; lane < 32; lane += 16) {
      auto value = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(result[reg] + lane));
      _mm512_storeu_si512(d[reg] + lane, value);
    }
}

} // namespace goc

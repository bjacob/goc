// SPDX-License-Identifier: MIT

#include "rdna4_swmmac_integer.h"
#include "x86_64/rdna4_alu_x86_64_v3.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {

void swmmac_integer_x86_64_v3(unsigned k, uint32_t exec_mask, bool clamp, uint32_t *const *d,
                              const SwmmacIntegerInputs &input) {
  uint32_t result[8][32];
  for (unsigned row = 0; row < 16; ++row)
    for (unsigned col = 0; col < 16; col += 8) {
      auto acc = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input.acc[row] + col));
      for (unsigned stage = 0; stage < 2; ++stage) {
        auto sum = _mm256_setzero_si256();
        for (unsigned local = 0; local < k / 4; ++local) {
          unsigned ck = swmmac_integer_ck(stage, local);
          auto b = _mm256_loadu_si256(
              reinterpret_cast<const __m256i *>(input.b[input.selected[row][ck]] + col));
          sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(_mm256_set1_epi32(input.a[row][ck]), b));
        }
        auto next = _mm256_add_epi32(acc, sum);
        if (clamp)
          next = saturate_signed_sum(acc, sum, next);
        acc = next;
      }
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[row % 8] + col + 16 * (row / 8)), acc);
    }
  for (unsigned reg = 0; reg < 8; ++reg)
    for (unsigned lane = 0; lane < 32; lane += 8) {
      auto value = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(result[reg] + lane));
      auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                              _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
      _mm256_maskstore_epi32(reinterpret_cast<int *>(d[reg] + lane), lane_exec_mask, value);
    }
}

} // namespace goc

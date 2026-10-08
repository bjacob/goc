// SPDX-License-Identifier: MIT

#include "x86_64/rdna4_half_fma_x86_64_v3.h"
#include "goc/goc.h"
#include "rdna4_fma.h"
#include "rdna4_half_fma.h"
#include "x86_64/rdna4_half_x86_64_v3.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {

template <FmaOperands Operands>
void half_fma_x86_64_v3(bool saturate, uint32_t mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                        const uint32_t *b, const uint32_t *c, uint16_t literal) {
  if constexpr (Operands != FmaOperands::Registers)
    mode &= GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_D;
  int a_shift = mode & GOC_ALU_HIGH_A ? 16 : 0;
  int b_shift = mode & GOC_ALU_HIGH_B ? 16 : 0;
  int c_shift = mode & GOC_ALU_HIGH_C ? 16 : 0;
  int d_shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  auto keep = _mm256_set1_epi32(d_shift ? 65535 : -65536);
  for (int lane = 0; lane < 32; lane += 8) {
    auto x = half_input<false>(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane)),
                               a_shift, mode);
    auto y = half_input<false>(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane)),
                               b_shift, mode >> 1);
    __m256 z;
    if constexpr (Operands == FmaOperands::MultiplyLiteral) {
      z = y;
      y = half_input<false>(_mm256_set1_epi32(literal), 0, 0);
    } else if constexpr (Operands == FmaOperands::AddLiteral) {
      z = half_input<false>(_mm256_set1_epi32(literal), 0, 0);
    } else {
      z = half_input<false>(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(c + lane)),
                            c_shift, mode >> 2);
    }
    auto result = half_fma_value(x, y, z, saturate, mode);
    result = _mm256_sll_epi32(result, _mm_cvtsi32_si128(d_shift));
    auto old = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d + lane));
    result = _mm256_or_si256(result, _mm256_and_si256(old, keep));
    auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(mask >> lane)),
                                    _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), active, result);
  }
}

template void half_fma_x86_64_v3<FmaOperands::Registers>(bool, uint32_t, uint32_t, uint32_t *,
                                                         const uint32_t *, const uint32_t *,
                                                         const uint32_t *, uint16_t);

template void half_fma_x86_64_v3<FmaOperands::MultiplyLiteral>(bool, uint32_t, uint32_t, uint32_t *,
                                                               const uint32_t *, const uint32_t *,
                                                               const uint32_t *, uint16_t);

template void half_fma_x86_64_v3<FmaOperands::AddLiteral>(bool, uint32_t, uint32_t, uint32_t *,
                                                          const uint32_t *, const uint32_t *,
                                                          const uint32_t *, uint16_t);

} // namespace goc

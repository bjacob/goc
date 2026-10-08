// SPDX-License-Identifier: MIT

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
  uint32_t omod = (mode >> 6) & 3;
  const float scales[] = {1, 2, 4, 0.5f};
  auto scale = _mm256_set1_ps(scales[omod]);
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
    // Half products are exact in FP32. TwoSum's residual gives the direction
    // needed for round-to-odd, avoiding double rounding on half-way cases.
    auto product = _mm256_mul_ps(x, y);
    auto value = _mm256_add_ps(product, z);
    auto virtual_c = _mm256_sub_ps(value, product);
    auto error = _mm256_add_ps(_mm256_sub_ps(product, _mm256_sub_ps(value, virtual_c)),
                               _mm256_sub_ps(z, virtual_c));
    auto bits = _mm256_castps_si256(value);
    auto even =
        _mm256_cmpeq_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(1)), _mm256_setzero_si256());
    auto adjust = _mm256_and_si256(
        even, _mm256_castps_si256(_mm256_cmp_ps(error, _mm256_setzero_ps(), _CMP_NEQ_OQ)));
    auto opposite = _mm256_srai_epi32(_mm256_xor_si256(bits, _mm256_castps_si256(error)), 31);
    auto step = _mm256_or_si256(opposite, _mm256_set1_epi32(1));
    bits = _mm256_add_epi32(bits, _mm256_and_si256(adjust, step));
    value = _mm256_castsi256_ps(bits);
    auto result = half_narrow<false>(value, saturate);
    if (omod) {
      auto magnitude = _mm256_and_si256(result, _mm256_set1_epi32(0x7fff));
      auto tiny = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x0400), magnitude);
      auto boundary = _mm256_cmpeq_epi32(magnitude, _mm256_set1_epi32(0x0400));
      auto abs_value = _mm256_and_ps(value, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)));
      auto below = _mm256_castps_si256(
          _mm256_cmp_ps(abs_value, _mm256_set1_ps(0x1p-14f - 0x1p-26f), _CMP_LT_OQ));
      tiny = _mm256_or_si256(tiny, _mm256_and_si256(boundary, below));
      auto scaled =
          half_narrow<false>(_mm256_mul_ps(half_input<false>(result, 0, 0), scale), saturate);
      if (omod == 3) {
        auto newly_tiny = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x0800), magnitude);
        scaled = _mm256_blendv_epi8(scaled, _mm256_and_si256(result, _mm256_set1_epi32(0x8000)),
                                    newly_tiny);
      }
      result = _mm256_andnot_si256(tiny, scaled);
    }
    if (mode & GOC_ALU_CLAMP) {
      auto negative = _mm256_cmpgt_epi32(result, _mm256_set1_epi32(0x7fff));
      auto nan = _mm256_cmpgt_epi32(_mm256_and_si256(result, _mm256_set1_epi32(0x7fff)),
                                    _mm256_set1_epi32(0x7c00));
      result = _mm256_andnot_si256(_mm256_or_si256(negative, nan),
                                   _mm256_min_epi32(result, _mm256_set1_epi32(0x3c00)));
    }
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

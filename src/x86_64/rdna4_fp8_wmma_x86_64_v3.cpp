// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_simd.h"
#include "x86_64/rdna4_fp8.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {
namespace {

template <bool Bf8> void decode(float (&values)[16][16], const uint32_t *const *v) {
  for (int k = 0; k < 16; ++k)
    for (int index = 0; index < 16; index += 8) {
      auto words = _mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(v[(k % 8) / 4] + index + 16 * (k / 8)));
      words = _mm256_srl_epi32(words, _mm_cvtsi32_si128(8 * (k % 4)));
      _mm256_storeu_ps(values[k] + index, widen_fp8<Bf8>(words));
    }
}

template <bool Bf8A, bool Bf8B>
void wmma(uint32_t modifiers, uint32_t *const *d, const uint32_t *const *a,
          const uint32_t *const *b, const uint32_t *const *c) {
  // Decode each packed factor once, then reuse across the full output matrix.
  alignas(32) float left[16][16], right[16][16];
  decode<Bf8A>(left, a);
  decode<Bf8B>(right, b);
  const auto keep = _mm256_set1_epi32(modifiers & GOC_WMMA_ABS_C ? INT32_MAX : -1);
  const auto flip = _mm256_set1_epi32(modifiers & GOC_WMMA_NEG_C ? INT32_MIN : 0);
  uint32_t result[8][32];
  for (int row = 0; row < 16; ++row) {
    const int reg = row % 8, lane = 16 * (row / 8);
    auto low = _mm256_castsi256_ps(_mm256_xor_si256(
        _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(c[reg] + lane)),
                         keep),
        flip));
    auto high = _mm256_castsi256_ps(_mm256_xor_si256(
        _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(c[reg] + lane + 8)),
                         keep),
        flip));
    for (int k = 0; k < 16; ++k) {
      auto x = _mm256_set1_ps(left[k][row]);
      low = _mm256_fmadd_ps(x, _mm256_load_ps(right[k]), low);
      high = _mm256_fmadd_ps(x, _mm256_load_ps(right[k] + 8), high);
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[reg] + lane), _mm256_castps_si256(low));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[reg] + lane + 8),
                        _mm256_castps_si256(high));
  }
  // Snapshot all inputs and results before writes, including cross-register aliases.
  for (int lane = 0; lane < 32; lane += 8) {
    for (int reg = 0; reg < 8; ++reg)
      _mm256_storeu_si256(
          reinterpret_cast<__m256i *>(d[reg] + lane),
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(result[reg] + lane)));
  }
}

} // namespace

void fp8_wmma_x86_64_v3(bool bf8_a, bool bf8_b, uint32_t modifiers, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c) {
  if (bf8_a) {
    if (bf8_b)
      wmma<true, true>(modifiers, d, a, b, c);
    else
      wmma<true, false>(modifiers, d, a, b, c);
  } else {
    if (bf8_b)
      wmma<false, true>(modifiers, d, a, b, c);
    else
      wmma<false, false>(modifiers, d, a, b, c);
  }
}

} // namespace goc

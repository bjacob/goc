// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "swmmac_float.h"
#include "x86_64/half_x86_64_v3.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {

namespace {

// Accumulate in compressed K order, narrowing packed results after each DOT4.
template <bool Bf16, bool Packed>
[[gnu::noinline]] void accumulate(uint32_t mode, SwmmacFloatInputs &input, bool saturate) {
  const uint32_t flip[] = {mode & GOC_WMMA_NEG_LO_B ? 0x80000000u : 0,
                           mode & GOC_WMMA_NEG_HI_B ? 0x80000000u : 0};
  for (unsigned row = 0; row < 16; ++row)
    for (unsigned col = 0; col < 16; col += 8) {
      auto acc = _mm256_loadu_ps(input.acc[row] + col);
      for (unsigned ck = 0; ck < 16; ++ck) {
        auto bv = _mm256_loadu_ps(input.b[input.selected[row][ck]] + col);
        bv = _mm256_xor_ps(bv, _mm256_castsi256_ps(_mm256_set1_epi32(int(flip[ck & 1]))));
        acc = _mm256_fmadd_ps(_mm256_set1_ps(input.a[row][ck]), bv, acc);
        if constexpr (Packed) {
          if ((ck + 1) % 4 == 0) {
            if constexpr (Bf16)
              acc = _mm256_castsi256_ps(
                  _mm256_and_si256(_mm256_castps_si256(acc), _mm256_set1_epi32(int(0xffff0000U))));
            else
              acc = half_input<false>(half_narrow<false>(acc, saturate), 0, 0);
          }
        }
      }
      _mm256_storeu_ps(input.acc[row] + col, acc);
    }
}

} // namespace

template <bool Bf16, bool Packed>
void swmmac_float_x86_64_v3(uint32_t mode, uint32_t *const *d, SwmmacFloatInputs &input,
                            bool saturate) {
  accumulate<Bf16, Packed>(mode, input, saturate);
  uint32_t result[Packed ? 4 : 8][32];
  swmmac_float_pack<Bf16, Packed>(result, input.acc, saturate);
  for (unsigned reg = 0; reg < (Packed ? 4u : 8u); ++reg)
    for (unsigned lane = 0; lane < 32; lane += 8) {
      auto value = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(result[reg] + lane));
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(d[reg] + lane), value);
    }
}

template void swmmac_float_x86_64_v3<false, false>(uint32_t, uint32_t *const *, SwmmacFloatInputs &,
                                                   bool);
template void swmmac_float_x86_64_v3<false, true>(uint32_t, uint32_t *const *, SwmmacFloatInputs &,
                                                  bool);
template void swmmac_float_x86_64_v3<true, false>(uint32_t, uint32_t *const *, SwmmacFloatInputs &,
                                                  bool);
template void swmmac_float_x86_64_v3<true, true>(uint32_t, uint32_t *const *, SwmmacFloatInputs &,
                                                 bool);

} // namespace goc

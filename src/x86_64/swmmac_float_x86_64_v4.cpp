// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "swmmac_float.h"

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
    for (unsigned col = 0; col < 16; col += 16) {
      auto acc = _mm512_loadu_ps(input.acc[row] + col);
      for (unsigned ck = 0; ck < 16; ++ck) {
        auto bv = _mm512_loadu_ps(input.b[input.selected[row][ck]] + col);
        bv = _mm512_xor_ps(bv, _mm512_castsi512_ps(_mm512_set1_epi32(int(flip[ck & 1]))));
        acc = _mm512_fmadd_ps(_mm512_set1_ps(input.a[row][ck]), bv, acc);
        if constexpr (Packed) {
          if ((ck + 1) % 4 == 0) {
            if constexpr (Bf16)
              acc = _mm512_castsi512_ps(
                  _mm512_and_si512(_mm512_castps_si512(acc), _mm512_set1_epi32(int(0xffff0000U))));
            else {
              auto half = _mm512_cvtps_ph(acc, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
              if (saturate) {
                auto bits = _mm512_cvtepu16_epi32(half);
                auto magnitude =
                    _mm512_and_si512(_mm512_castps_si512(acc), _mm512_set1_epi32(INT32_MAX));
                auto finite = _mm512_cmplt_epi32_mask(magnitude, _mm512_set1_epi32(0x7f800000));
                auto infinity = _mm512_cmpeq_epi32_mask(
                    _mm512_and_si512(bits, _mm512_set1_epi32(0x7fff)), _mm512_set1_epi32(0x7c00));
                bits = _mm512_mask_sub_epi32(bits, finite & infinity, bits, _mm512_set1_epi32(1));
                half = _mm512_cvtepi32_epi16(bits);
              }
              acc = _mm512_cvtph_ps(half);
            }
          }
        }
      }
      _mm512_storeu_ps(input.acc[row] + col, acc);
    }
}

} // namespace

template <bool Bf16, bool Packed>
void swmmac_float_x86_64_v4(uint32_t mode, uint32_t *const *d, SwmmacFloatInputs &input,
                            bool saturate) {
  accumulate<Bf16, Packed>(mode, input, saturate);
  uint32_t result[Packed ? 4 : 8][32];
  swmmac_float_pack<Bf16, Packed>(result, input.acc, saturate);
  for (unsigned reg = 0; reg < (Packed ? 4u : 8u); ++reg)
    for (unsigned lane = 0; lane < 32; lane += 16) {
      auto value = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(result[reg] + lane));
      _mm512_storeu_si512(d[reg] + lane, value);
    }
}

template void swmmac_float_x86_64_v4<false, false>(uint32_t, uint32_t *const *, SwmmacFloatInputs &,
                                                   bool);
template void swmmac_float_x86_64_v4<false, true>(uint32_t, uint32_t *const *, SwmmacFloatInputs &,
                                                  bool);
template void swmmac_float_x86_64_v4<true, false>(uint32_t, uint32_t *const *, SwmmacFloatInputs &,
                                                  bool);
template void swmmac_float_x86_64_v4<true, true>(uint32_t, uint32_t *const *, SwmmacFloatInputs &,
                                                 bool);

} // namespace goc

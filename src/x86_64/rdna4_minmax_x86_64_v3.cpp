// SPDX-License-Identifier: MIT

#include "x86_64/rdna4_minmax_x86_64_v3.h"
#include "goc/goc.h"
#include "rdna4_minmax.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {
namespace {

template <bool FirstMaximum, bool SecondMaximum, bool Propagate>
void run(uint32_t mask, uint32_t mode, uint32_t *d, const uint32_t *a, const uint32_t *b,
         const uint32_t *c) {
  const auto ka = _mm256_set1_epi32(mode & GOC_ALU_ABS_A ? INT32_MAX : -1);
  const auto kb = _mm256_set1_epi32(mode & GOC_ALU_ABS_B ? INT32_MAX : -1);
  const auto kc = _mm256_set1_epi32(mode & GOC_ALU_ABS_C ? INT32_MAX : -1);
  const auto na = _mm256_set1_epi32(mode & GOC_ALU_NEG_A ? INT32_MIN : 0);
  const auto nb = _mm256_set1_epi32(mode & GOC_ALU_NEG_B ? INT32_MIN : 0);
  const auto nc = _mm256_set1_epi32(mode & GOC_ALU_NEG_C ? INT32_MIN : 0);
  const float scales[] = {1, 2, 4, 0.5f};
  const auto scale = _mm256_set1_ps(scales[(mode >> 6) & 3]);
  for (int lane = 0; lane < 32; lane += 8) {
    auto x = _mm256_castsi256_ps(_mm256_xor_si256(
        _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane)), ka), na));
    auto y = _mm256_castsi256_ps(_mm256_xor_si256(
        _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane)), kb), nb));
    auto z = _mm256_castsi256_ps(_mm256_xor_si256(
        _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(c + lane)), kc), nc));
    auto ab = minmax<FirstMaximum, Propagate>(x, y);
    auto value = minmax<SecondMaximum, Propagate>(ab, z);
    if (mode & GOC_ALU_OMOD_HALF)
      value = _mm256_mul_ps(value, scale);
    if (mode & GOC_ALU_CLAMP)
      value = _mm256_min_ps(_mm256_max_ps(value, _mm256_setzero_ps()), _mm256_set1_ps(1));
    auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(mask >> lane)),
                                    _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), active, _mm256_castps_si256(value));
  }
}

} // namespace

void minmax3_x86_64_v3(Minmax3 op, uint32_t mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                       const uint32_t *b, const uint32_t *c) {
  switch (op) {
  case Minmax3::Min3Num:
    return run<false, false, false>(mask, mode, d, a, b, c);
  case Minmax3::Max3Num:
    return run<true, true, false>(mask, mode, d, a, b, c);
  case Minmax3::MinmaxNum:
    return run<false, true, false>(mask, mode, d, a, b, c);
  case Minmax3::MaxminNum:
    return run<true, false, false>(mask, mode, d, a, b, c);
  case Minmax3::Minimum3:
    return run<false, false, true>(mask, mode, d, a, b, c);
  case Minmax3::Maximum3:
    return run<true, true, true>(mask, mode, d, a, b, c);
  case Minmax3::MinimumMaximum:
    return run<false, true, true>(mask, mode, d, a, b, c);
  case Minmax3::MaximumMinimum:
    return run<true, false, true>(mask, mode, d, a, b, c);
  }
}

} // namespace goc

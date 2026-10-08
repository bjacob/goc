// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_binary.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {
namespace {

// Correct the x86 selection rules for NaNs and opposite-signed zeros.
template <bool Maximum, bool Propagate> __m256 minmax(__m256 x, __m256 y) {
  auto a = _mm256_castps_si256(x), b = _mm256_castps_si256(y);
  auto magnitude = _mm256_set1_epi32(INT32_MAX);
  auto infinity = _mm256_set1_epi32(0x7f800000);
  auto quiet = _mm256_set1_epi32(0x00400000);
  auto an = _mm256_cmpgt_epi32(_mm256_and_si256(a, magnitude), infinity);
  auto bn = _mm256_cmpgt_epi32(_mm256_and_si256(b, magnitude), infinity);
  auto value = Maximum ? _mm256_max_ps(x, y) : _mm256_min_ps(x, y);
  auto equal = _mm256_cmp_ps(x, y, _CMP_EQ_OQ);
  auto tie = Maximum ? _mm256_and_si256(a, b) : _mm256_or_si256(a, b);
  value = _mm256_blendv_ps(value, _mm256_castsi256_ps(tie), equal);
  if constexpr (Propagate) {
    auto snb = _mm256_andnot_si256(_mm256_cmpeq_epi32(_mm256_and_si256(b, quiet), quiet), bn);
    auto sna = _mm256_andnot_si256(_mm256_cmpeq_epi32(_mm256_and_si256(a, quiet), quiet), an);
    auto selected = _mm256_blendv_epi8(b, a, an);
    selected = _mm256_blendv_epi8(selected, b, snb);
    selected = _mm256_blendv_epi8(selected, a, sna);
    value = _mm256_blendv_ps(value, _mm256_castsi256_ps(_mm256_or_si256(selected, quiet)),
                             _mm256_castsi256_ps(_mm256_or_si256(an, bn)));
  } else {
    value = _mm256_blendv_ps(value, x, _mm256_castsi256_ps(bn));
    value = _mm256_blendv_ps(value, y, _mm256_castsi256_ps(an));
    value = _mm256_blendv_ps(value, _mm256_castsi256_ps(_mm256_or_si256(a, quiet)),
                             _mm256_castsi256_ps(_mm256_and_si256(an, bn)));
  }
  return value;
}

template <Binary Op>
void run(uint32_t mask, uint32_t mode, uint32_t *d, const uint32_t *a, const uint32_t *b) {
  const auto ka = _mm256_set1_epi32(mode & GOC_ALU_ABS_A ? INT32_MAX : -1);
  const auto kb = _mm256_set1_epi32(mode & GOC_ALU_ABS_B ? INT32_MAX : -1);
  const auto na = _mm256_set1_epi32(mode & GOC_ALU_NEG_A ? INT32_MIN : 0);
  const auto nb = _mm256_set1_epi32(mode & GOC_ALU_NEG_B ? INT32_MIN : 0);
  const float scales[] = {1, 2, 4, 0.5f};
  const auto scale = _mm256_set1_ps(scales[(mode >> 6) & 3]);
  for (int lane = 0; lane < 32; lane += 8) {
    auto x = _mm256_castsi256_ps(_mm256_xor_si256(
        _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane)), ka), na));
    auto y = _mm256_castsi256_ps(_mm256_xor_si256(
        _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane)), kb), nb));
    __m256 value;
    if constexpr (Op == Binary::Add)
      value = _mm256_add_ps(x, y);
    if constexpr (Op == Binary::Sub)
      value = _mm256_sub_ps(x, y);
    if constexpr (Op == Binary::Subrev)
      value = _mm256_sub_ps(y, x);
    if constexpr (Op == Binary::Mul)
      value = _mm256_mul_ps(x, y);
    if constexpr (Op == Binary::MinNum)
      value = minmax<false, false>(x, y);
    if constexpr (Op == Binary::MaxNum)
      value = minmax<true, false>(x, y);
    if constexpr (Op == Binary::Minimum)
      value = minmax<false, true>(x, y);
    if constexpr (Op == Binary::Maximum)
      value = minmax<true, true>(x, y);
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

void binary_x86_64_v3(Binary op, uint32_t mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                      const uint32_t *b) {
  switch (op) {
  case Binary::Add:
    return run<Binary::Add>(mask, mode, d, a, b);
  case Binary::Sub:
    return run<Binary::Sub>(mask, mode, d, a, b);
  case Binary::Subrev:
    return run<Binary::Subrev>(mask, mode, d, a, b);
  case Binary::MinNum:
    return run<Binary::MinNum>(mask, mode, d, a, b);
  case Binary::MaxNum:
    return run<Binary::MaxNum>(mask, mode, d, a, b);
  case Binary::Minimum:
    return run<Binary::Minimum>(mask, mode, d, a, b);
  case Binary::Maximum:
    return run<Binary::Maximum>(mask, mode, d, a, b);
  case Binary::Mul:
    return run<Binary::Mul>(mask, mode, d, a, b);
  }
}

} // namespace goc

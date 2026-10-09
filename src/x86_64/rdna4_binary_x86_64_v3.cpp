// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_binary.h"
#include "x86_64/rdna4_alu_x86_64_v3.h"
#include "x86_64/rdna4_minmax_x86_64_v3.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {
namespace {

template <Binary Op>
void run(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a, const uint32_t *b) {
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
    if constexpr (Op == Binary::MulDx9Zero) {
      auto zero = _mm256_setzero_ps();
      auto has_zero =
          _mm256_or_ps(_mm256_cmp_ps(x, zero, _CMP_EQ_OQ), _mm256_cmp_ps(y, zero, _CMP_EQ_OQ));
      value = _mm256_andnot_ps(has_zero, _mm256_mul_ps(x, y));
    }
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
    if (mode & GOC_ALU_OMOD_HALF) {
      value = prepare_omod_f32(value, mode);
      value = _mm256_mul_ps(value, scale);
    }
    if (mode & GOC_ALU_CLAMP)
      value = _mm256_min_ps(_mm256_max_ps(value, _mm256_setzero_ps()), _mm256_set1_ps(1));
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                            _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask,
                           _mm256_castps_si256(value));
  }
}

} // namespace

void binary_x86_64_v3(Binary op, uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                      const uint32_t *b) {
  switch (op) {
  case Binary::Add:
    return run<Binary::Add>(exec_mask, mode, d, a, b);
  case Binary::Sub:
    return run<Binary::Sub>(exec_mask, mode, d, a, b);
  case Binary::Subrev:
    return run<Binary::Subrev>(exec_mask, mode, d, a, b);
  case Binary::MinNum:
    return run<Binary::MinNum>(exec_mask, mode, d, a, b);
  case Binary::MaxNum:
    return run<Binary::MaxNum>(exec_mask, mode, d, a, b);
  case Binary::Minimum:
    return run<Binary::Minimum>(exec_mask, mode, d, a, b);
  case Binary::Maximum:
    return run<Binary::Maximum>(exec_mask, mode, d, a, b);
  case Binary::MulDx9Zero:
    return run<Binary::MulDx9Zero>(exec_mask, mode, d, a, b);
  case Binary::Mul:
    return run<Binary::Mul>(exec_mask, mode, d, a, b);
  }
}

} // namespace goc

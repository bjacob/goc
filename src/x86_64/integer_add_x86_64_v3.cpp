// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "integer_add.h"
#include "x86_64/alu_x86_64_v3.h"

#include <immintrin.h>
#include <stdint.h>

namespace goc {

template <IntegerAdd Op, bool Signed>
void integer_add_sat_x86_64_v3(uint32_t exec_mask, uint32_t *d, const uint32_t *a,
                               const uint32_t *b) {
  if constexpr (Op == IntegerAdd::Subrev) {
    const auto *temporary = a;
    a = b;
    b = temporary;
  }
  for (int lane = 0; lane < 32; lane += 8) {
    auto x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane)),
         y = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
    __m256i result;
    if constexpr (Op == IntegerAdd::Add)
      result = _mm256_add_epi32(x, y);
    else
      result = _mm256_sub_epi32(x, y);
    if constexpr (Signed && Op == IntegerAdd::Add) {
      result = saturate_signed_sum(x, y, result);
    } else if constexpr (Signed) {
      auto different_inputs = _mm256_xor_si256(x, y);
      auto different_result = _mm256_xor_si256(x, result);
      __m256i overflow;
      // Subtraction overflows when opposite-sign inputs flip A's sign.
      overflow = _mm256_and_si256(different_inputs, different_result);
      overflow = _mm256_srai_epi32(overflow, 31);
      auto saturated = _mm256_xor_si256(_mm256_set1_epi32(INT32_MAX), _mm256_srai_epi32(x, 31));
      result = _mm256_or_si256(_mm256_and_si256(overflow, saturated),
                               _mm256_andnot_si256(overflow, result));
    } else {
      auto bias = _mm256_set1_epi32(INT32_MIN);
      auto ordered_x = _mm256_xor_si256(x, bias);
      if constexpr (Op == IntegerAdd::Add) {
        auto overflow = _mm256_cmpgt_epi32(ordered_x, _mm256_xor_si256(result, bias));
        result = _mm256_or_si256(result, overflow);
      } else {
        auto borrow = _mm256_cmpgt_epi32(_mm256_xor_si256(y, bias), ordered_x);
        result = _mm256_andnot_si256(borrow, result);
      }
    }
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                            _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask, result);
  }
}

template void integer_add_sat_x86_64_v3<IntegerAdd::Add, false>(uint32_t, uint32_t *,
                                                                const uint32_t *, const uint32_t *);
template void integer_add_sat_x86_64_v3<IntegerAdd::Sub, false>(uint32_t, uint32_t *,
                                                                const uint32_t *, const uint32_t *);
template void integer_add_sat_x86_64_v3<IntegerAdd::Subrev, false>(uint32_t, uint32_t *,
                                                                   const uint32_t *,
                                                                   const uint32_t *);
template void integer_add_sat_x86_64_v3<IntegerAdd::Add, true>(uint32_t, uint32_t *,
                                                               const uint32_t *, const uint32_t *);
template void integer_add_sat_x86_64_v3<IntegerAdd::Sub, true>(uint32_t, uint32_t *,
                                                               const uint32_t *, const uint32_t *);

} // namespace goc

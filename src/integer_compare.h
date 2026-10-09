// SPDX-License-Identifier: MIT

#ifndef GOC_INTEGER_COMPARE_H_
#define GOC_INTEGER_COMPARE_H_

#include <stdint.h>

namespace goc {

template <unsigned Predicate> uint32_t integer_compare_result(uint32_t less, uint32_t equal) {
  if constexpr (Predicate == 1)
    return less;
  if constexpr (Predicate == 2)
    return equal;
  if constexpr (Predicate == 3)
    return less | equal;
  if constexpr (Predicate == 4)
    return ~(less | equal);
  if constexpr (Predicate == 5)
    return ~equal;
  return ~less;
}

template <unsigned Bits, bool Signed, unsigned Predicate>
uint32_t integer_compare_x86_64_v3(uint32_t mode, const uint32_t *const *a,
                                   const uint32_t *const *b);
template <unsigned Bits, bool Signed, unsigned Predicate>
uint32_t integer_compare_x86_64_v4(uint32_t mode, const uint32_t *const *a,
                                   const uint32_t *const *b);

} // namespace goc

#endif

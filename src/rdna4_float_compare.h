// SPDX-License-Identifier: MIT

#ifndef GOC_RDNA4_FLOAT_COMPARE_H_
#define GOC_RDNA4_FLOAT_COMPARE_H_

#include "rdna4_integer_compare.h"

#include <stdint.h>

namespace goc {

template <unsigned Predicate>
uint32_t float_compare_result(uint32_t less, uint32_t equal, uint32_t ordered) {
  constexpr unsigned base = Predicate >= 8 ? 15 - Predicate : Predicate;
  uint32_t holds;
  if constexpr (base == 7)
    holds = ordered;
  else
    holds = integer_compare_result<base>(less, equal) & ordered;
  if constexpr (Predicate >= 8)
    return ~holds;
  return holds;
}

template <unsigned Bits, unsigned Predicate>
uint32_t float_compare_x86_64_v3(uint32_t mode, bool flush, const uint32_t *const *a,
                                 const uint32_t *const *b);
template <unsigned Bits, unsigned Predicate>
uint32_t float_compare_x86_64_v4(uint32_t mode, bool flush, const uint32_t *const *a,
                                 const uint32_t *const *b);

} // namespace goc

#endif

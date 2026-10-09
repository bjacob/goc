// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <stdint.h>

namespace goc_test {

using CarryFn = decltype(&goc_rdna4_v_add_co_ci_u32);

inline int carry_add(uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                     uint32_t *carry, const uint32_t *const *a, const uint32_t *const *b,
                     uint32_t) {
  return goc_rdna4_v_add_co_u32(flags, mask, mode, d, carry, a, b);
}

inline int carry_sub(uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                     uint32_t *carry, const uint32_t *const *a, const uint32_t *const *b,
                     uint32_t) {
  return goc_rdna4_v_sub_co_u32(flags, mask, mode, d, carry, a, b);
}

inline int carry_subrev(uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                        uint32_t *carry, const uint32_t *const *a, const uint32_t *const *b,
                        uint32_t) {
  return goc_rdna4_v_subrev_co_u32(flags, mask, mode, d, carry, a, b);
}

static const CarryFn carry_functions[] = {carry_add,
                                          carry_sub,
                                          carry_subrev,
                                          goc_rdna4_v_add_co_ci_u32,
                                          goc_rdna4_v_sub_co_ci_u32,
                                          goc_rdna4_v_subrev_co_ci_u32};

struct CarryResult {
  uint32_t value;
  bool carry;
};

// Compute one lane with 64-bit unsigned arithmetic; operation order is add,
// sub, subrev, then the same three with an input carry/borrow bit.
inline CarryResult carry_reference(unsigned op, uint32_t a, uint32_t b, bool input_carry,
                                   bool clamp) {
  uint64_t first = op % 3 == 2 ? b : a, second = op % 3 == 2 ? a : b;
  second += op >= 3 && input_carry;
  bool carry;
  uint64_t value;
  if (op % 3 == 0) {
    value = first + second;
    carry = value > UINT32_MAX;
  } else {
    value = first - second;
    carry = first < second;
  }
  if (clamp && carry)
    value = op % 3 == 0 ? UINT32_MAX : 0;
  return {uint32_t(value), carry};
}

} // namespace goc_test

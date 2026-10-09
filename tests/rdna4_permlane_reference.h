// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"
#include <stdint.h>

namespace goc_test {
inline int permlane_call(unsigned op, uint64_t flags, uint64_t mask, uint64_t mode,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                         uint32_t lo, uint32_t hi) {
  switch (op) {
  case 0:
    return goc_rdna4_v_permlane16_b32(flags, mask, mode, d, a, lo, hi);
  case 1:
    return goc_rdna4_v_permlanex16_b32(flags, mask, mode, d, a, lo, hi);
  case 2:
    return goc_rdna4_v_permlane16_var_b32(flags, mask, mode, d, a, b);
  default:
    return goc_rdna4_v_permlanex16_var_b32(flags, mask, mode, d, a, b);
  }
}

inline uint32_t permlane_reference(unsigned op, uint32_t mask, uint32_t mode, unsigned lane,
                                   const uint32_t *a, const uint32_t *b, uint32_t old, uint32_t lo,
                                   uint32_t hi) {
  if (!(mask & (1u << lane)))
    return old;
  unsigned index = op >= 2 ? b[lane] & 15 : ((lane % 16 < 8 ? lo : hi) >> (4 * (lane % 8))) & 15;
  unsigned row = lane / 16;
  if (op & 1)
    row = 1 - row;
  unsigned src = row * 16 + index;
  if ((mask & (1u << src)) || (mode & GOC_PERMLANE_FI))
    return a[src];
  return mode & GOC_PERMLANE_BOUND_CTRL ? 0 : old;
}
} // namespace goc_test

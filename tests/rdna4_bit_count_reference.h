// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <stdint.h>

namespace goc_test {

inline uint32_t bit_count_reference(int op, uint32_t a, uint32_t b, unsigned lane) {
  if (op < 3) {
    uint32_t count = 0;
    bool sign = op == 2 && (a >> 31);
    for (unsigned bit = 0; bit < 32; ++bit) {
      unsigned position = op == 1 ? bit : 31 - bit;
      if (bool((a >> position) & 1) != sign)
        return count;
      ++count;
    }
    return UINT32_MAX;
  }
  uint32_t count = 0;
  for (unsigned bit = 0; bit < 32; ++bit) {
    bool selected = op == 3 || ((op == 4 || op == 6) ? bit < lane : bit + 32 < lane);
    if (selected && ((a >> bit) & 1))
      ++count;
  }
  return b + count;
}

inline int count_leading(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *) {
  return goc_rdna4_v_clz_i32_u32(flags, mask, mode, d, a);
}

inline int count_trailing(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *) {
  return goc_rdna4_v_ctz_i32_b32(flags, mask, mode, d, a);
}

inline int count_sign(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
                      const uint32_t *const *a, const uint32_t *const *) {
  return goc_rdna4_v_cls_i32(flags, mask, mode, d, a);
}

} // namespace goc_test

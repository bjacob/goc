// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <stdint.h>

namespace goc_test {

inline uint32_t boolean_mode(int op, unsigned selection) {
  return op < 4 ? 0
                : (selection & 1 ? GOC_ALU_HIGH_A : 0) |
                      (selection & 2 && op != 7 ? GOC_ALU_HIGH_B : 0) |
                      (selection & 4 ? GOC_ALU_HIGH_D : 0);
}

// Independent per-bit truth table, retaining the unselected destination bits.
inline uint32_t boolean_reference(int op, uint32_t a, uint32_t b, uint32_t d, uint32_t mode) {
  uint32_t result = op < 4 ? 0 : d;
  unsigned sa = mode & GOC_ALU_HIGH_A ? 16 : 0;
  unsigned sb = mode & GOC_ALU_HIGH_B ? 16 : 0;
  unsigned sd = mode & GOC_ALU_HIGH_D ? 16 : 0;
  for (unsigned bit = 0; bit < (op < 4 ? 32u : 16u); ++bit) {
    bool x = (a >> (sa + bit)) & 1, y = (b >> (sb + bit)) & 1;
    bool value = false;
    switch (op % 4) {
    case 0:
      value = x && y;
      break;
    case 1:
      value = x || y;
      break;
    case 2:
      value = x != y;
      break;
    case 3:
      value = !x;
      break;
    }
    uint32_t mask = uint32_t(1) << (sd + bit);
    result = (result & ~mask) | (value ? mask : 0);
  }
  return result;
}

inline int boolean_not32(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *) {
  return goc_rdna4_v_not_b32(flags, mask, mode, d, a);
}

inline int boolean_not16(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *) {
  return goc_rdna4_v_not_b16(flags, mask, mode, d, a);
}

} // namespace goc_test

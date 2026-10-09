// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <stdint.h>

namespace goc_test {

inline uint32_t cndmask_mode(unsigned m) {
  return (m & 1 ? GOC_ALU_ABS_A : 0) | (m & 2 ? GOC_ALU_ABS_B : 0) | (m & 4 ? GOC_ALU_NEG_A : 0) |
         (m & 8 ? GOC_ALU_NEG_B : 0) | (m & 16 ? GOC_ALU_HIGH_A : 0) |
         (m & 32 ? GOC_ALU_HIGH_B : 0) | (m & 64 ? GOC_ALU_HIGH_D : 0);
}

inline uint32_t cndmask_reference(bool half, uint32_t a, uint32_t b, uint32_t old, unsigned m,
                                  bool choose) {
  uint32_t source = choose ? b : a;
  unsigned source_offset = half && (m & (16u << unsigned(choose))) ? 16 : 0;
  unsigned dest_offset = half && (m & 64) ? 16 : 0;
  unsigned width = half ? 16 : 32;
  uint32_t result = half ? old : 0;
  for (unsigned bit = 0; bit < width; ++bit) {
    bool value = (source >> (source_offset + bit)) & 1;
    if (bit == width - 1) {
      if (m & (1u << unsigned(choose)))
        value = false;
      if (m & (4u << unsigned(choose)))
        value = !value;
    }
    uint32_t position = uint32_t(1) << (dest_offset + bit);
    result = (result & ~position) | (value ? position : 0);
  }
  return result;
}

} // namespace goc_test

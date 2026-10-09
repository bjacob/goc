// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"
#include "mixed_fma_reference.h"

#include <stdint.h>

namespace goc_test {

inline uint32_t interp16_mode(unsigned op, unsigned m, unsigned wait = 0) {
  return (m & 7) | (m & 8 ? GOC_ALU_CLAMP : 0) | (m & 16 ? GOC_ALU_HIGH_A : 0) |
         (m & 32 ? (op & 1 ? GOC_ALU_HIGH_D : GOC_ALU_HIGH_C) : 0) |
         (wait << GOC_INTERP_WAIT_EXP_SHIFT);
}

inline uint32_t interp16_reference(unsigned op, const uint32_t *a, const uint32_t *b,
                                   const uint32_t *c, uint32_t old, unsigned lane, unsigned m,
                                   bool saturate) {
  unsigned quad = lane / 4 * 4;
  uint32_t mode =
      (m & 7) | (m & 8 ? GOC_ALU_CLAMP : 0) | (m & 16 ? GOC_ALU_HIGH_A : 0) | GOC_MIX_F16_A;
  if (!(op & 1))
    mode |= GOC_MIX_F16_C | (m & 32 ? GOC_ALU_HIGH_C : 0);
  return mixed_fma_reference::evaluate(op & 1 ? (m & 32 ? 2 : 1) : 0, a[quad + (op & 1 ? 2 : 1)],
                                       b[lane], c[op & 1 ? lane : quad], old, mode, saturate,
                                       op >= 2);
}

inline uint32_t interp16_canonical(unsigned op, unsigned m, uint32_t word) {
  if (!(op & 1))
    return (word & 0x7fffffff) > 0x7f800000 ? 0x7fc00000 : word;
  unsigned shift = m & 32 ? 16 : 0;
  if (((word >> shift) & 0x7fff) > 0x7c00)
    word = (word & ~(65535u << shift)) | (0x7e00u << shift);
  return word;
}

inline void interp16_capture_inputs(uint32_t words[4][32], unsigned start) {
  for (unsigned lane = 0; lane < 32; ++lane) {
    unsigned i = start + lane;
    words[0][lane] = (i * 0x7395a831u) ^ 0xa7925163u;
    words[1][lane] = (i * 0x83a1459du) ^ 0x5389d241u;
    words[2][lane] = (i * 0x38459317u) ^ 0x187abd31u;
    words[3][lane] = 0xcafebeef;
  }
}

} // namespace goc_test

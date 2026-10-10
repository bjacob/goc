// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc_test {

// Operand corpus used by capture_unary_exact.py; op 3 selects EXP boundaries.
inline uint32_t unary_capture_input(unsigned op, unsigned i) {
  const uint32_t edge[] = {0,          0x80000000, 1,          0x80000001, 0x7fffff,   0x807fffff,
                           0x800000,   0x80800000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000,
                           0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7f800001, 0xff800001,
                           0x7fc12345, 0xffc12345, 0x7e800000, 0xfe800000, 0x7e7fffff, 0x7e800001,
                           0x00800001, 0x00ffffff, 0x01000000, 0x7e000000, 0x7dffffff, 0x7e000001,
                           0x3f7fffff, 0x3f800001};
  if (op == 3 && i >= 20 && i < 28) {
    const uint32_t exp_edge[] = {0x43000000, 0x42ffffff, 0xc2fc0000, 0xc2fc0001,
                                 0x33800000, 0x337fffff, 0xb3800000, 0xb37fffff};
    return exp_edge[i - 20];
  }
  if (i < 32)
    return edge[i];
  if (op == 3 && i < 4096)
    return ((i & 1) << 31) | (0x33800000 + ((i * 0x7395a831U) % (0x43000000 - 0x33800000)));
  if (i < 4096)
    return (i - 32) * 0x9e3779b9U;
  return ((i - 4096) & 1 ? 0x80000000 : 0) | ((((i - 4096) >> 1) % 254 + 1) << 23) |
         ((i * 0x7395a831U) & 0x7fffff);
}

} // namespace goc_test

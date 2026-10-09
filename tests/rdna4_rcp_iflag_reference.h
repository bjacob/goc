// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <cmath>
#include <cstring>
#include <stdint.h>

namespace goc_test {

inline uint32_t rcp_iflag_input(unsigned i) {
  const uint32_t edges[] = {0,          0x80000000, 1,          0x80000001, 0x007fffff, 0x807fffff,
                            0x00800000, 0x80800000, 0x3f800000, 0xbf800000, 0x7f800000, 0xff800000,
                            0x7f800001, 0x7fc00001, 0x7f7fffff, 0xff7fffff, 0x40000000, 0xc0000000,
                            0x3f000000, 0xbf000000, 0x7e800000, 0xfe800000, 0x7e800001, 0xfe800001,
                            0x7e7fffff, 0xfe7fffff, 0x3f800001, 0xbf800001, 0x00800001, 0x80800001,
                            0x3eaaaaab, 0xbeaaaaab};
  unsigned wave = i / 32, lane = i % 32;
  if (wave < 32)
    return edges[wave];
  if (wave < 64)
    return lane == wave - 32 ? (wave < 48 ? 0 : 1) : 0x3f800000;
  return (i * 0x7395a831u) ^ 0xa7925163u;
}

inline uint32_t rcp_iflag_mode(unsigned m) {
  return (m & 1 ? GOC_ALU_ABS_A : 0) | (m & 2 ? GOC_ALU_NEG_A : 0) | (((m >> 2) & 3) << 6) |
         (m & 16 ? GOC_ALU_CLAMP : 0);
}

inline uint32_t rcp_iflag_reference(uint32_t raw, unsigned m) {
  float input;
  std::memcpy(&input, &raw, 4);
  double x = input;
  if (m & 1)
    x = std::fabs(x);
  if (m & 2)
    x = -x;
  if (std::fabs(x) < 0x1p-126)
    x = std::copysign(0., x);
  float rounded = float(1.0 / x);
  if (std::fabs(rounded) < 0x1p-126f)
    rounded = std::copysign(0.f, rounded);
  if (m & 12) {
    if (rounded == 0)
      rounded = 0;
    rounded = std::ldexp(rounded, ((m >> 2) & 3) == 3 ? -1 : int((m >> 2) & 3));
  }
  if (m & 16)
    rounded = !(rounded > 0) ? 0 : rounded > 1 ? 1 : rounded;
  if (std::fabs(rounded) < 0x1p-126f)
    rounded = std::copysign(0.f, rounded);
  uint32_t bits;
  std::memcpy(&bits, &rounded, 4);
  return bits;
}

inline bool rcp_iflag_close(uint32_t a, uint32_t b) {
  uint32_t am = a & 0x7fffffff, bm = b & 0x7fffffff;
  if (am > 0x7f800000)
    return bm > 0x7f800000;
  if (!am || !bm || am >= 0x7f800000 || bm >= 0x7f800000)
    return a == b;
  return a > b ? a - b <= 2 : b - a <= 2;
}

inline uint32_t rcp_iflag_status(const uint32_t *a, uint32_t mask, unsigned m, uint32_t initial) {
  if (!(m & 16))
    for (unsigned lane = 0; lane < 32; ++lane)
      if (((mask >> lane) & 1) && (a[lane] & 0x7fffffff) < 0x00800000)
        initial |= 0x40;
  return initial;
}

} // namespace goc_test

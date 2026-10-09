// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc_test {

// RX 9070, gfx1201, RNE. All four hardware denormal modes produced
// identical results for 4 datasets x 512 modifiers x 32 lanes (262,144 words).
// FNV-1a hashes use little-endian bytes, with all NaNs normalized to 0x7fc00000.
static const uint64_t dx9_hardware_hashes[] = {0x6861d7131cc26bc5ULL, 0xebe85366f7780f45ULL,
                                               0xb07008694d13afb5ULL, 0xef878f0125b62325ULL};

inline void dx9_hardware_inputs(unsigned set, unsigned lane, uint32_t &a, uint32_t &b,
                                uint32_t &c) {
  static const uint32_t values[] = {
      0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000,
      0x80800000, 0x00ffffff, 0x80ffffff, 0x01000000, 0x81000000, 0x3f000000, 0xbf000000,
      0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x7e800000, 0xfe800000, 0x7f000000,
      0xff000000, 0x7f7fffff, 0xff7fffff, 0x3f800001, 0x3f7ffffe, 0xbf800000, 0x7f800000,
      0xff800000, 0x7fc12345, 0xffc12345, 0x7f812345};
  a = values[lane];
  b = 0x3f800000;
  c = 0x80000000;
  if (set == 1) {
    a = lane & 1 ? 0x80000000 : 0;
    b = values[(lane * 5 + 3) % 32];
    c = values[lane];
  }
  if (set == 2) {
    b = values[(lane * 7 + 1) % 32];
    c = values[(lane * 3 + 5) % 32];
  }
  if (set == 3) {
    a = 0x00800000 + lane % 8;
    b = 0x3efffff0 + lane;
    c = lane & 1 ? 0x80000000 : 0;
  }
}

} // namespace goc_test

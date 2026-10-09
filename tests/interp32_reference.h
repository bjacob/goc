// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <cmath>
#include <cstring>
#include <stdint.h>

namespace goc_test {

inline float interp32_float(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, 4);
  return value;
}

inline uint32_t interp32_bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, 4);
  return bits;
}

inline uint32_t interp32_mode(unsigned m, unsigned wait = 0) {
  return (m & 7) | (m & 8 ? GOC_ALU_CLAMP : 0) | (wait << GOC_INTERP_WAIT_EXP_SHIFT);
}

inline float interp32_reference(bool p2, const uint32_t *a, const uint32_t *b, const uint32_t *c,
                                unsigned lane, unsigned m) {
  unsigned quad = lane / 4 * 4;
  long double x = interp32_float(a[quad + (p2 ? 2 : 1)]), y = interp32_float(b[lane]),
              z = interp32_float(c[p2 ? lane : quad]);
  if (m & 1)
    x = -x;
  if (m & 2)
    y = -y;
  if (m & 4)
    z = -z;
  float value = float(std::fma(x, y, z));
  if (m & 8)
    value = !(value > 0) ? 0 : value > 1 ? 1 : value;
  return value;
}

inline void interp32_capture_inputs(uint32_t words[4][32]) {
  for (unsigned lane = 0; lane < 32; ++lane) {
    words[0][lane] = interp32_bits(float(int((lane * 13 + 7) % 64) - 32) * 0.125f);
    words[1][lane] = interp32_bits(float(int((lane * 29 + 3) % 64) - 32) * 0.0625f);
    words[2][lane] = interp32_bits(float(int((lane * 37 + 17) % 64) - 32) * 0.03125f);
    words[3][lane] = 0xdeadbeef;
  }
}

} // namespace goc_test

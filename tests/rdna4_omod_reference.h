// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <stdint.h>

namespace goc_test {

inline float omod_f32_reference(float value, uint32_t mode) {
  unsigned omod = (mode >> 6) & 3;
  if (!omod)
    return value;
  float magnitude = std::abs(value);
  if (magnitude < std::ldexp(1.0f, -126))
    return 0.0f;
  if (omod == 3 && magnitude < std::ldexp(1.0f, -125))
    return std::copysign(0.0f, value);
  return std::ldexp(value, omod == 3 ? -1 : int(omod));
}

} // namespace goc_test

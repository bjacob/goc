// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"
#include "internal.h"

#include <algorithm>
#include <cmath>
#include <stdint.h>

namespace goc_test {
namespace minmax_reference {

inline bool nan(uint32_t x) { return (x & 0x7fffffff) > 0x7f800000; }

// Sortable integer keys distinguish signed zeros without host min/max rules.
inline uint32_t ordered(uint32_t x) { return x & 0x80000000 ? ~x : x ^ 0x80000000; }

inline uint32_t select(uint32_t a, uint32_t b, bool maximum, bool propagate) {
  if (propagate) {
    for (uint32_t x : {a, b})
      if (nan(x) && !(x & 0x00400000))
        return x | 0x00400000;
    for (uint32_t x : {a, b})
      if (nan(x))
        return x;
  } else {
    if (nan(a) && nan(b))
      return a | 0x00400000;
    if (nan(a) || nan(b))
      return nan(a) ? b : a;
  }
  return (maximum ? ordered(a) > ordered(b) : ordered(a) < ordered(b)) ? a : b;
}

inline uint32_t reference(int op, uint32_t a, uint32_t b, uint32_t c, uint32_t mode) {
  uint32_t inputs[] = {a, b, c};
  for (int i = 0; i < 3; ++i) {
    if (mode & (GOC_ALU_ABS_A << i))
      inputs[i] &= 0x7fffffff;
    if (mode & (GOC_ALU_NEG_A << i))
      inputs[i] ^= 0x80000000;
  }
  bool first_max = op % 4 == 1 || op % 4 == 3;
  bool second_max = op % 4 == 1 || op % 4 == 2;
  uint32_t result =
      select(select(inputs[0], inputs[1], first_max, op >= 4), inputs[2], second_max, op >= 4);
  if (op == 8) {
    if (nan(inputs[0]) || nan(inputs[1]) || nan(inputs[2])) {
      result = select(select(inputs[0], inputs[1], false, false), inputs[2], false, false);
    } else {
      // Sorting independently establishes the median for nonzero values. The
      // ISA's first-maximum removal rule additionally defines signed-zero ties.
      uint32_t sorted[] = {inputs[0], inputs[1], inputs[2]};
      std::sort(sorted, sorted + 3, [](uint32_t a, uint32_t b) { return ordered(a) < ordered(b); });
      result = sorted[1];
      if ((result & 0x7fffffff) == 0) {
        float maximum = goc::as_float(sorted[2]);
        int drop = goc::as_float(inputs[0]) == maximum   ? 0
                   : goc::as_float(inputs[1]) == maximum ? 1
                                                         : 2;
        result = select(inputs[(drop + 1) % 3], inputs[(drop + 2) % 3], true, false);
      }
    }
  }
  const int exponents[] = {0, 1, 2, -1};
  unsigned scale = (mode >> 6) & 3;
  float value = goc::as_float(result);
  if (scale && std::abs(value) < std::ldexp(1.0f, -126))
    value = 0;
  else if (scale == 3 && std::abs(value) < std::ldexp(1.0f, -125))
    value = std::copysign(0.0f, value);
  value = std::ldexp(value, exponents[scale]);
  if (mode & GOC_ALU_CLAMP)
    value = !(value > 0) ? 0 : std::min(value, 1.0f);
  return goc::as_bits(value);
}

} // namespace minmax_reference
} // namespace goc_test

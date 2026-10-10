// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"
#include "half_reference.h"

#include <algorithm>
#include <cmath>
#include <stdint.h>

namespace goc_test {

inline uint16_t half_exponent_reference(int op, uint32_t a, uint32_t b, uint32_t mode,
                                        bool saturate) {
  double x = goc_test::half_value(uint16_t(a >> (mode & GOC_ALU_HIGH_A ? 16 : 0)));
  if (op == 1) {
    int exponent = 0;
    if (x != 0 && std::isfinite(x))
      std::frexp(x, &exponent);
    return uint16_t(exponent);
  }
  if (mode & GOC_ALU_ABS_A)
    x = std::abs(x);
  if (mode & GOC_ALU_NEG_A)
    x = -x;
  uint16_t raw = uint16_t(b >> (mode & GOC_ALU_HIGH_B ? 16 : 0));
  int exponent = raw <= INT16_MAX ? int(raw) : int(raw) - 65536;
  double result = std::ldexp(x, exponent);
  // Preserve finite-overflow provenance through subsequent output scaling.
  if (std::isfinite(x) && std::abs(result) > 1e100)
    result = std::copysign(1e100, result);
  const double scales[] = {1, 2, 4, 0.5};
  if (mode & GOC_ALU_OMOD_HALF) {
    if (std::abs(result) < 0x1p-14)
      result = 0;
    else if ((mode & GOC_ALU_OMOD_HALF) == GOC_ALU_OMOD_HALF && std::abs(result) < 0x1p-13)
      result = std::copysign(0.0, result);
  }
  if (mode & GOC_ALU_OMOD_HALF)
    result = goc_test::half_value(goc_test::half_bits(result, saturate));
  result *= scales[(mode >> 6) & 3];
  if (mode & GOC_ALU_CLAMP)
    result = !(result > 0) ? 0 : std::min(result, 1.0);
  return goc_test::half_bits(result, saturate);
}

} // namespace goc_test

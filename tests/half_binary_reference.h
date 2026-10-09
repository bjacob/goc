// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"
#include "half_reference.h"

#include <algorithm>
#include <cmath>
#include <stdint.h>

namespace goc_test {

using HalfBinaryFn = decltype(&goc_v_add_f16);
const HalfBinaryFn half_binary_functions[] = {
    goc_v_add_f16,     goc_v_sub_f16,     goc_v_subrev_f16,  goc_v_mul_f16,
    goc_v_min_num_f16, goc_v_max_num_f16, goc_v_minimum_f16, goc_v_maximum_f16};

const uint32_t half_binary_known = GOC_ALU_NEG_A | GOC_ALU_NEG_B | GOC_ALU_ABS_A | GOC_ALU_ABS_B |
                                   GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP | GOC_ALU_HIGH_A |
                                   GOC_ALU_HIGH_B | GOC_ALU_HIGH_D;

inline uint16_t half_binary_reference(int op, uint32_t a, uint32_t b, uint32_t mode,
                                      bool saturate) {
  double x = goc_test::half_value(uint16_t(a >> (mode & GOC_ALU_HIGH_A ? 16 : 0)));
  double y = goc_test::half_value(uint16_t(b >> (mode & GOC_ALU_HIGH_B ? 16 : 0)));
  if (mode & GOC_ALU_ABS_A)
    x = std::abs(x);
  if (mode & GOC_ALU_ABS_B)
    y = std::abs(y);
  if (mode & GOC_ALU_NEG_A)
    x = -x;
  if (mode & GOC_ALU_NEG_B)
    y = -y;
  double result = goc_test::half_binary(op, x, y);
  const double scales[] = {1, 2, 4, 0.5};
  if (mode & GOC_ALU_OMOD_HALF) {
    if (std::abs(result) < 0x1p-14)
      result = 0;
    else if ((mode & GOC_ALU_OMOD_HALF) == GOC_ALU_OMOD_HALF && std::abs(result) < 0x1p-13)
      result = std::copysign(0.0, result);
  }
  result *= scales[(mode >> 6) & 3];
  if (mode & GOC_ALU_CLAMP)
    result = !(result > 0) ? 0 : std::min(result, 1.0);
  return goc_test::half_bits(result, saturate);
}

inline uint32_t half_binary_modifiers(unsigned variant) {
  return (variant & 3) | ((variant & 12) << 1) | ((variant & 112) << 2) |
         (variant & 128 ? GOC_ALU_HIGH_A : 0) | (variant & 256 ? GOC_ALU_HIGH_B : 0) |
         (variant & 512 ? GOC_ALU_HIGH_D : 0);
}

const char *const half_binary_names[] = {"v_add_f16",     "v_sub_f16",     "v_subrev_f16",
                                         "v_mul_f16",     "v_min_num_f16", "v_max_num_f16",
                                         "v_minimum_f16", "v_maximum_f16"};

} // namespace goc_test

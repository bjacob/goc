// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"
#include "half_reference.h"

#include <algorithm>
#include <cmath>
#include <stdint.h>

namespace goc_test {

using HalfUnaryFn = decltype(&goc_v_trunc_f16);
const HalfUnaryFn half_unary_functions[] = {goc_v_trunc_f16, goc_v_ceil_f16,      goc_v_rndne_f16,
                                            goc_v_floor_f16, goc_v_sqrt_f16,      goc_v_rcp_f16,
                                            goc_v_rsq_f16,   goc_v_exp_f16,       goc_v_log_f16,
                                            goc_v_fract_f16, goc_v_frexp_mant_f16};
const uint32_t half_unary_known = GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF |
                                  GOC_ALU_CLAMP | GOC_ALU_HIGH_A | GOC_ALU_HIGH_D;

inline uint16_t half_unary_reference(int op, uint32_t input, uint32_t mode, bool saturate) {
  double x = goc_test::half_value(uint16_t(input >> (mode & GOC_ALU_HIGH_A ? 16 : 0)));
  if (mode & GOC_ALU_ABS_A)
    x = std::abs(x);
  if (mode & GOC_ALU_NEG_A)
    x = -x;
  double result = goc_test::half_unary(op, x);
  if (op == 7 || op == 8) {
    if (op == 8 && x == 0 && saturate)
      result = -65504;
    result = goc_test::half_value(goc_test::half_bits(result, saturate));
  }
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

inline uint32_t half_unary_modifiers(unsigned variant) {
  return (variant & 1 ? GOC_ALU_NEG_A : 0) | (variant & 2 ? GOC_ALU_ABS_A : 0) |
         ((variant & 28) << 4) | (variant & 32 ? GOC_ALU_HIGH_A : 0) |
         (variant & 64 ? GOC_ALU_HIGH_D : 0);
}

const char *const half_unary_names[] = {
    "v_trunc_f16", "v_ceil_f16", "v_rndne_f16", "v_floor_f16", "v_sqrt_f16",      "v_rcp_f16",
    "v_rsq_f16",   "v_exp_f16",  "v_log_f16",   "v_fract_f16", "v_frexp_mant_f16"};

} // namespace goc_test

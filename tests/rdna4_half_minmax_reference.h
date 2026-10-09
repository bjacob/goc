// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"
#include "rdna4_half_reference.h"

#include <algorithm>
#include <cmath>
#include <stdint.h>

namespace goc_test {

using HalfMinmaxFn = decltype(&goc_rdna4_v_min3_num_f16);
const HalfMinmaxFn half_minmax_functions[] = {
    goc_rdna4_v_min3_num_f16,       goc_rdna4_v_max3_num_f16,       goc_rdna4_v_minmax_num_f16,
    goc_rdna4_v_maxmin_num_f16,     goc_rdna4_v_minimum3_f16,       goc_rdna4_v_maximum3_f16,
    goc_rdna4_v_minimummaximum_f16, goc_rdna4_v_maximumminimum_f16, goc_rdna4_v_med3_num_f16};

inline bool half_minmax_nan(uint16_t bits) { return (bits & 0x7fff) > 0x7c00; }

inline uint16_t half_minmax_ordered(uint16_t bits) {
  return bits & 0x8000 ? uint16_t(~bits) : bits ^ 0x8000;
}

inline uint16_t half_minmax_select(uint16_t a, uint16_t b, bool maximum, bool propagate) {
  if (half_minmax_nan(a) || half_minmax_nan(b)) {
    if (propagate || (half_minmax_nan(a) && half_minmax_nan(b)))
      return 0x7e00;
    return half_minmax_nan(a) ? b : a;
  }
  return (maximum ? half_minmax_ordered(a) > half_minmax_ordered(b)
                  : half_minmax_ordered(a) < half_minmax_ordered(b))
             ? a
             : b;
}

inline uint16_t half_minmax_reference(int op, uint32_t a, uint32_t b, uint32_t c, uint32_t mode,
                                      bool saturate) {
  const uint32_t words[] = {a, b, c};
  uint16_t input[3];
  for (int i = 0; i < 3; ++i) {
    input[i] = uint16_t(words[i] >> (mode & (GOC_ALU_HIGH_A << i) ? 16 : 0));
    if (mode & (GOC_ALU_ABS_A << i))
      input[i] &= 0x7fff;
    if (mode & (GOC_ALU_NEG_A << i))
      input[i] ^= 0x8000;
  }
  bool first_maximum = op % 4 == 1 || op % 4 == 3;
  bool second_maximum = op % 4 == 1 || op % 4 == 2;
  uint16_t result =
      half_minmax_select(half_minmax_select(input[0], input[1], first_maximum, op >= 4), input[2],
                         second_maximum, op >= 4);
  if (op == 8) {
    if (half_minmax_nan(input[0]) || half_minmax_nan(input[1]) || half_minmax_nan(input[2])) {
      result = half_minmax_select(half_minmax_select(input[0], input[1], false, false), input[2],
                                  false, false);
    } else {
      uint16_t sorted[] = {input[0], input[1], input[2]};
      std::sort(sorted, sorted + 3, [](uint16_t a, uint16_t b) {
        return half_minmax_ordered(a) < half_minmax_ordered(b);
      });
      result = sorted[1];
    }
  }
  const double scales[] = {1, 2, 4, 0.5};
  double value = goc_test::half_value(result);
  if (mode & GOC_ALU_OMOD_HALF) {
    if (std::abs(value) < 0x1p-14)
      value = 0;
    else if ((mode & GOC_ALU_OMOD_HALF) == GOC_ALU_OMOD_HALF && std::abs(value) < 0x1p-13)
      value = std::copysign(0.0, value);
  }
  value *= scales[(mode >> 6) & 3];
  if (mode & GOC_ALU_CLAMP)
    value = !(value > 0) ? 0 : std::min(value, 1.0);
  return goc_test::half_bits(value, saturate);
}

const char *const half_minmax_names[] = {
    "v_min3_num_f16",       "v_max3_num_f16",       "v_minmax_num_f16",
    "v_maxmin_num_f16",     "v_minimum3_f16",       "v_maximum3_f16",
    "v_minimummaximum_f16", "v_maximumminimum_f16", "v_med3_num_f16"};

} // namespace goc_test

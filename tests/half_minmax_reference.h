// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"
#include "half_reference.h"

#include <algorithm>
#include <cmath>
#include <stdint.h>

namespace goc_test {

using HalfMinmaxFn = decltype(&goc_v_min3_num_f16);
const HalfMinmaxFn half_minmax_functions[] = {
    goc_v_min3_num_f16,       goc_v_max3_num_f16,       goc_v_minmax_num_f16,
    goc_v_maxmin_num_f16,     goc_v_minimum3_f16,       goc_v_maximum3_f16,
    goc_v_minimummaximum_f16, goc_v_maximumminimum_f16, goc_v_med3_num_f16};

inline bool half_minmax_nan(uint16_t bits) { return (bits & 0x7fff) > 0x7c00; }

inline uint16_t half_minmax_ordered(uint16_t bits) {
  return bits & 0x8000 ? uint16_t(~bits) : bits ^ 0x8000;
}

inline uint16_t half_minmax_select(uint16_t a, uint16_t b, bool maximum, bool propagate) {
  if (half_minmax_nan(a) || half_minmax_nan(b)) {
    if (propagate) {
      for (uint16_t bits : {a, b})
        if (half_minmax_nan(bits) && !(bits & 0x200))
          return bits | 0x200;
      return half_minmax_nan(a) ? a : b;
    }
    if (half_minmax_nan(a) && half_minmax_nan(b))
      return a | 0x200;
    return half_minmax_nan(a) ? b : a;
  }
  return (maximum ? half_minmax_ordered(a) > half_minmax_ordered(b)
                  : half_minmax_ordered(a) < half_minmax_ordered(b))
             ? a
             : b;
}

inline uint16_t half_minmax_reference(int op, uint32_t a, uint32_t b, uint32_t c, uint32_t mode,
                                      bool saturate, bool rdna3 = false, uint64_t flags = 0) {
  auto select = [rdna3, flags](uint16_t x, uint16_t y, bool maximum, bool propagate) {
    if (rdna3 && (flags & GOC_FP_IEEE_MODE)) {
      for (uint16_t bits : {x, y})
        if (half_minmax_nan(bits) && !(bits & 0x200))
          return uint16_t(bits | 0x200);
    } else if (rdna3 && half_minmax_nan(x) && half_minmax_nan(y)) {
      return x;
    }
    return half_minmax_select(x, y, maximum, propagate);
  };
  if (rdna3 && (!(flags & GOC_FP_FLUSH_OUTPUT_DENORMALS) || (flags & GOC_FP_IEEE_MODE)))
    mode &= ~GOC_ALU_OMOD_HALF;
  const uint32_t words[] = {a, b, c};
  uint16_t input[3];
  for (int i = 0; i < 3; ++i) {
    input[i] = uint16_t(words[i] >> (mode & (GOC_ALU_HIGH_A << i) ? 16 : 0));
    if (mode & (GOC_ALU_ABS_A << i))
      input[i] &= 0x7fff;
    if (mode & (GOC_ALU_NEG_A << i))
      input[i] ^= 0x8000;
    if ((flags & GOC_FP_FLUSH_INPUT_DENORMALS) && (input[i] & 0x7fff) < 0x400)
      input[i] &= 0x8000;
  }
  bool first_maximum = op % 4 == 1 || op % 4 == 3;
  bool second_maximum = op % 4 == 1 || op % 4 == 2;
  uint16_t result =
      select(select(input[0], input[1], first_maximum, op >= 4), input[2], second_maximum, op >= 4);
  if (op == 8) {
    if (half_minmax_nan(input[0]) || half_minmax_nan(input[1]) || half_minmax_nan(input[2])) {
      result = select(select(input[0], input[1], false, false), input[2], false, false);
    } else {
      uint16_t sorted[] = {input[0], input[1], input[2]};
      std::sort(sorted, sorted + 3, [](uint16_t a, uint16_t b) {
        return half_minmax_ordered(a) < half_minmax_ordered(b);
      });
      result = sorted[1];
      if (rdna3 && (result & 0x7fff) == 0) {
        double maximum = goc_test::half_value(sorted[2]);
        int drop = goc_test::half_value(input[0]) == maximum   ? 0
                   : goc_test::half_value(input[1]) == maximum ? 1
                                                               : 2;
        result = select(input[(drop + 1) % 3], input[(drop + 2) % 3], true, false);
      }
    }
  }
  // Preserve payloads without converting through the numeric double reference.
  // Active host-arithmetic OMOD quiets a surviving signaling NaN in loose mode.
  if (half_minmax_nan(result)) {
    if (mode & GOC_ALU_CLAMP)
      return 0;
    return mode & GOC_ALU_OMOD_HALF ? uint16_t(result | 0x200) : result;
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

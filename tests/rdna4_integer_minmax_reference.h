// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <algorithm>
#include <stdint.h>

namespace goc_test {

inline const char *const integer_minmax_names[] = {
    "v_min_i32",    "v_max_i32",    "v_min3_i32",   "v_max3_i32", "v_minmax_i32",
    "v_maxmin_i32", "v_med3_i32",   "v_min_u32",    "v_max_u32",  "v_min3_u32",
    "v_max3_u32",   "v_minmax_u32", "v_maxmin_u32", "v_med3_u32"};

using IntegerMinmaxFn = decltype(&goc_rdna4_v_fma_f32);

template <auto Function>
int integer_minmax_binary(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *) {
  return Function(flags, mask, mode, d, a, b);
}

inline const IntegerMinmaxFn integer_minmax_functions[] = {
    integer_minmax_binary<goc_rdna4_v_min_i32>,
    integer_minmax_binary<goc_rdna4_v_max_i32>,
    goc_rdna4_v_min3_i32,
    goc_rdna4_v_max3_i32,
    goc_rdna4_v_minmax_i32,
    goc_rdna4_v_maxmin_i32,
    goc_rdna4_v_med3_i32,
    integer_minmax_binary<goc_rdna4_v_min_u32>,
    integer_minmax_binary<goc_rdna4_v_max_u32>,
    goc_rdna4_v_min3_u32,
    goc_rdna4_v_max3_u32,
    goc_rdna4_v_minmax_u32,
    goc_rdna4_v_maxmin_u32,
    goc_rdna4_v_med3_u32};

inline uint32_t integer_minmax_reference(int instruction, uint32_t a, uint32_t b, uint32_t c) {
  int64_t values[] = {a, b, c};
  if (instruction < 7)
    for (auto &value : values)
      if (value > INT32_MAX)
        value -= INT64_C(4294967296);
  int op = instruction % 7;
  std::sort(values, values + 2);
  if (op == 0 || op == 1)
    return uint32_t(values[op]);
  if (op == 4)
    return uint32_t(values[0] > values[2] ? values[0] : values[2]);
  if (op == 5)
    return uint32_t(values[1] < values[2] ? values[1] : values[2]);
  std::sort(values, values + 3);
  return uint32_t(values[op == 2 ? 0 : op == 3 ? 2 : 1]);
}

} // namespace goc_test

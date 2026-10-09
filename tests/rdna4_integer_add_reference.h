// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <stdint.h>

namespace goc_test {

inline const char *const integer_add_names[] = {"v_add_nc_u32", "v_sub_nc_u32", "v_subrev_nc_u32",
                                                "v_add_nc_i32", "v_sub_nc_i32", "v_add3_u32"};

using IntegerAddFn = decltype(&goc_rdna4_v_add3_u32);

template <auto Function>
int integer_add_binary(uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *) {
  return Function(flags, mask, mode, d, a, b);
}

inline const IntegerAddFn integer_add_functions[] = {
    integer_add_binary<goc_rdna4_v_add_nc_u32>,    integer_add_binary<goc_rdna4_v_sub_nc_u32>,
    integer_add_binary<goc_rdna4_v_subrev_nc_u32>, integer_add_binary<goc_rdna4_v_add_nc_i32>,
    integer_add_binary<goc_rdna4_v_sub_nc_i32>,    goc_rdna4_v_add3_u32};

inline uint32_t integer_add_reference(int op, uint32_t a, uint32_t b, uint32_t c, bool clamp) {
  int64_t x = a, y = b;
  bool signed_op = op == 3 || op == 4;
  if (signed_op) {
    if (x > INT32_MAX)
      x -= INT64_C(4294967296);
    if (y > INT32_MAX)
      y -= INT64_C(4294967296);
  }
  int64_t result = op == 1 || op == 4 ? x - y : op == 2 ? y - x : x + y;
  if (op == 5)
    result += c;
  if (clamp) {
    int64_t low = signed_op ? INT32_MIN : 0, high = signed_op ? INT32_MAX : int64_t(UINT32_MAX);
    if (result < low)
      result = low;
    if (result > high)
      result = high;
  }
  return uint32_t(result);
}

} // namespace goc_test

// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <stdint.h>

namespace goc_test {

using IntegerCompareFn = decltype(&goc_rdna4_v_cmp_lt_i16);
static const IntegerCompareFn integer_compare_functions[] = {
    goc_rdna4_v_cmp_lt_i16,  goc_rdna4_v_cmp_eq_i16,  goc_rdna4_v_cmp_le_i16,
    goc_rdna4_v_cmp_gt_i16,  goc_rdna4_v_cmp_ne_i16,  goc_rdna4_v_cmp_ge_i16,
    goc_rdna4_v_cmpx_lt_i16, goc_rdna4_v_cmpx_eq_i16, goc_rdna4_v_cmpx_le_i16,
    goc_rdna4_v_cmpx_gt_i16, goc_rdna4_v_cmpx_ne_i16, goc_rdna4_v_cmpx_ge_i16,
    goc_rdna4_v_cmp_lt_u16,  goc_rdna4_v_cmp_eq_u16,  goc_rdna4_v_cmp_le_u16,
    goc_rdna4_v_cmp_gt_u16,  goc_rdna4_v_cmp_ne_u16,  goc_rdna4_v_cmp_ge_u16,
    goc_rdna4_v_cmpx_lt_u16, goc_rdna4_v_cmpx_eq_u16, goc_rdna4_v_cmpx_le_u16,
    goc_rdna4_v_cmpx_gt_u16, goc_rdna4_v_cmpx_ne_u16, goc_rdna4_v_cmpx_ge_u16,
    goc_rdna4_v_cmp_lt_i32,  goc_rdna4_v_cmp_eq_i32,  goc_rdna4_v_cmp_le_i32,
    goc_rdna4_v_cmp_gt_i32,  goc_rdna4_v_cmp_ne_i32,  goc_rdna4_v_cmp_ge_i32,
    goc_rdna4_v_cmpx_lt_i32, goc_rdna4_v_cmpx_eq_i32, goc_rdna4_v_cmpx_le_i32,
    goc_rdna4_v_cmpx_gt_i32, goc_rdna4_v_cmpx_ne_i32, goc_rdna4_v_cmpx_ge_i32,
    goc_rdna4_v_cmp_lt_u32,  goc_rdna4_v_cmp_eq_u32,  goc_rdna4_v_cmp_le_u32,
    goc_rdna4_v_cmp_gt_u32,  goc_rdna4_v_cmp_ne_u32,  goc_rdna4_v_cmp_ge_u32,
    goc_rdna4_v_cmpx_lt_u32, goc_rdna4_v_cmpx_eq_u32, goc_rdna4_v_cmpx_le_u32,
    goc_rdna4_v_cmpx_gt_u32, goc_rdna4_v_cmpx_ne_u32, goc_rdna4_v_cmpx_ge_u32,
    goc_rdna4_v_cmp_lt_i64,  goc_rdna4_v_cmp_eq_i64,  goc_rdna4_v_cmp_le_i64,
    goc_rdna4_v_cmp_gt_i64,  goc_rdna4_v_cmp_ne_i64,  goc_rdna4_v_cmp_ge_i64,
    goc_rdna4_v_cmpx_lt_i64, goc_rdna4_v_cmpx_eq_i64, goc_rdna4_v_cmpx_le_i64,
    goc_rdna4_v_cmpx_gt_i64, goc_rdna4_v_cmpx_ne_i64, goc_rdna4_v_cmpx_ge_i64,
    goc_rdna4_v_cmp_lt_u64,  goc_rdna4_v_cmp_eq_u64,  goc_rdna4_v_cmp_le_u64,
    goc_rdna4_v_cmp_gt_u64,  goc_rdna4_v_cmp_ne_u64,  goc_rdna4_v_cmp_ge_u64,
    goc_rdna4_v_cmpx_lt_u64, goc_rdna4_v_cmpx_eq_u64, goc_rdna4_v_cmpx_le_u64,
    goc_rdna4_v_cmpx_gt_u64, goc_rdna4_v_cmpx_ne_u64, goc_rdna4_v_cmpx_ge_u64,
};
static const char *const integer_compare_names[] = {
    "v_cmp_lt_i16",  "v_cmp_eq_i16",  "v_cmp_le_i16",  "v_cmp_gt_i16",  "v_cmp_ne_i16",
    "v_cmp_ge_i16",  "v_cmpx_lt_i16", "v_cmpx_eq_i16", "v_cmpx_le_i16", "v_cmpx_gt_i16",
    "v_cmpx_ne_i16", "v_cmpx_ge_i16", "v_cmp_lt_u16",  "v_cmp_eq_u16",  "v_cmp_le_u16",
    "v_cmp_gt_u16",  "v_cmp_ne_u16",  "v_cmp_ge_u16",  "v_cmpx_lt_u16", "v_cmpx_eq_u16",
    "v_cmpx_le_u16", "v_cmpx_gt_u16", "v_cmpx_ne_u16", "v_cmpx_ge_u16", "v_cmp_lt_i32",
    "v_cmp_eq_i32",  "v_cmp_le_i32",  "v_cmp_gt_i32",  "v_cmp_ne_i32",  "v_cmp_ge_i32",
    "v_cmpx_lt_i32", "v_cmpx_eq_i32", "v_cmpx_le_i32", "v_cmpx_gt_i32", "v_cmpx_ne_i32",
    "v_cmpx_ge_i32", "v_cmp_lt_u32",  "v_cmp_eq_u32",  "v_cmp_le_u32",  "v_cmp_gt_u32",
    "v_cmp_ne_u32",  "v_cmp_ge_u32",  "v_cmpx_lt_u32", "v_cmpx_eq_u32", "v_cmpx_le_u32",
    "v_cmpx_gt_u32", "v_cmpx_ne_u32", "v_cmpx_ge_u32", "v_cmp_lt_i64",  "v_cmp_eq_i64",
    "v_cmp_le_i64",  "v_cmp_gt_i64",  "v_cmp_ne_i64",  "v_cmp_ge_i64",  "v_cmpx_lt_i64",
    "v_cmpx_eq_i64", "v_cmpx_le_i64", "v_cmpx_gt_i64", "v_cmpx_ne_i64", "v_cmpx_ge_i64",
    "v_cmp_lt_u64",  "v_cmp_eq_u64",  "v_cmp_le_u64",  "v_cmp_gt_u64",  "v_cmp_ne_u64",
    "v_cmp_ge_u64",  "v_cmpx_lt_u64", "v_cmpx_eq_u64", "v_cmpx_le_u64", "v_cmpx_gt_u64",
    "v_cmpx_ne_u64", "v_cmpx_ge_u64",
};

inline uint32_t integer_compare_mode(unsigned m) {
  return (m & 1 ? GOC_ALU_HIGH_A : 0) | (m & 2 ? GOC_ALU_HIGH_B : 0);
}

inline void integer_compare_inputs(unsigned fmt, unsigned i, uint32_t *w) {
  unsigned bits = fmt == 0 ? 16 : fmt == 1 ? 32 : 64;
  uint64_t sign = uint64_t(1) << (bits - 1), max = sign + (sign - 1);
  uint64_t edges[] = {0,          1,          2,          sign - 2,      sign - 1, sign,
                      sign + 1,   sign + 2,   max - 1,    max,           0xffff,   0x10000,
                      0x7fffffff, 0x80000000, 0xffffffff, 0x100000000ULL};
  uint64_t a, b;
  if (i < 256) {
    a = edges[i % 16] & max;
    b = edges[i / 16] & max;
  } else {
    a = (uint64_t((i * 0x7395a831u) ^ 0xa7925163u) << 32) | ((i * 0x83a1459du) ^ 0x5389d241u);
    b = (uint64_t((i * 0x38459317u) ^ 0x187abd31u) << 32) | ((i * 0x9e3779b9u) ^ 0xa5a59669u);
    if (i % 4 == 0)
      b = a;
    else if (i % 4 == 1)
      b = (a & 0xffffffff00000000ULL) | uint32_t(b);
    else if (i % 4 == 2)
      b = (b & 0xffffffff00000000ULL) | uint32_t(a);
    if (fmt == 0) {
      a = i | ((65535 - i) << 16);
      b = (i % 4 == 0) ? a : ((i * 0x7395a831u) ^ 0xa7925163u);
    }
  }
  w[0] = uint32_t(a);
  w[1] = uint32_t(a >> 32);
  w[2] = uint32_t(b);
  w[3] = uint32_t(b >> 32);
}

inline bool integer_compare_reference(unsigned op, unsigned m, const uint32_t *w) {
  unsigned fmt = op / 24, predicate = op % 6;
  bool is_signed = !(op / 12 % 2);
  uint64_t a = fmt == 0   ? ((w[0] >> (m & 1 ? 16 : 0)) & 65535)
               : fmt == 1 ? w[0]
                          : (uint64_t(w[1]) << 32) | w[0];
  uint64_t b = fmt == 0   ? ((w[2] >> (m & 2 ? 16 : 0)) & 65535)
               : fmt == 1 ? w[2]
                          : (uint64_t(w[3]) << 32) | w[2];
  uint64_t sign = UINT64_C(1) << (fmt == 0 ? 15 : fmt == 1 ? 31 : 63);
  bool less = is_signed && ((a ^ b) & sign) ? bool(a & sign) : a < b;
  switch (predicate) {
  case 0:
    return less;
  case 1:
    return a == b;
  case 2:
    return less || a == b;
  case 3:
    return !less && a != b;
  case 4:
    return a != b;
  default:
    return !less;
  }
}

} // namespace goc_test

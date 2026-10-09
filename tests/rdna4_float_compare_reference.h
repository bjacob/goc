// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <stdint.h>

namespace goc_test {

using FloatCompareFn = decltype(&goc_rdna4_v_cmp_lt_f16);
static const FloatCompareFn float_compare_functions[] = {
    goc_rdna4_v_cmp_lt_f16,   goc_rdna4_v_cmp_eq_f16,   goc_rdna4_v_cmp_le_f16,
    goc_rdna4_v_cmp_gt_f16,   goc_rdna4_v_cmp_lg_f16,   goc_rdna4_v_cmp_ge_f16,
    goc_rdna4_v_cmp_o_f16,    goc_rdna4_v_cmp_u_f16,    goc_rdna4_v_cmp_nge_f16,
    goc_rdna4_v_cmp_nlg_f16,  goc_rdna4_v_cmp_ngt_f16,  goc_rdna4_v_cmp_nle_f16,
    goc_rdna4_v_cmp_neq_f16,  goc_rdna4_v_cmp_nlt_f16,  goc_rdna4_v_cmpx_lt_f16,
    goc_rdna4_v_cmpx_eq_f16,  goc_rdna4_v_cmpx_le_f16,  goc_rdna4_v_cmpx_gt_f16,
    goc_rdna4_v_cmpx_lg_f16,  goc_rdna4_v_cmpx_ge_f16,  goc_rdna4_v_cmpx_o_f16,
    goc_rdna4_v_cmpx_u_f16,   goc_rdna4_v_cmpx_nge_f16, goc_rdna4_v_cmpx_nlg_f16,
    goc_rdna4_v_cmpx_ngt_f16, goc_rdna4_v_cmpx_nle_f16, goc_rdna4_v_cmpx_neq_f16,
    goc_rdna4_v_cmpx_nlt_f16, goc_rdna4_v_cmp_lt_f32,   goc_rdna4_v_cmp_eq_f32,
    goc_rdna4_v_cmp_le_f32,   goc_rdna4_v_cmp_gt_f32,   goc_rdna4_v_cmp_lg_f32,
    goc_rdna4_v_cmp_ge_f32,   goc_rdna4_v_cmp_o_f32,    goc_rdna4_v_cmp_u_f32,
    goc_rdna4_v_cmp_nge_f32,  goc_rdna4_v_cmp_nlg_f32,  goc_rdna4_v_cmp_ngt_f32,
    goc_rdna4_v_cmp_nle_f32,  goc_rdna4_v_cmp_neq_f32,  goc_rdna4_v_cmp_nlt_f32,
    goc_rdna4_v_cmpx_lt_f32,  goc_rdna4_v_cmpx_eq_f32,  goc_rdna4_v_cmpx_le_f32,
    goc_rdna4_v_cmpx_gt_f32,  goc_rdna4_v_cmpx_lg_f32,  goc_rdna4_v_cmpx_ge_f32,
    goc_rdna4_v_cmpx_o_f32,   goc_rdna4_v_cmpx_u_f32,   goc_rdna4_v_cmpx_nge_f32,
    goc_rdna4_v_cmpx_nlg_f32, goc_rdna4_v_cmpx_ngt_f32, goc_rdna4_v_cmpx_nle_f32,
    goc_rdna4_v_cmpx_neq_f32, goc_rdna4_v_cmpx_nlt_f32, goc_rdna4_v_cmp_lt_f64,
    goc_rdna4_v_cmp_eq_f64,   goc_rdna4_v_cmp_le_f64,   goc_rdna4_v_cmp_gt_f64,
    goc_rdna4_v_cmp_lg_f64,   goc_rdna4_v_cmp_ge_f64,   goc_rdna4_v_cmp_o_f64,
    goc_rdna4_v_cmp_u_f64,    goc_rdna4_v_cmp_nge_f64,  goc_rdna4_v_cmp_nlg_f64,
    goc_rdna4_v_cmp_ngt_f64,  goc_rdna4_v_cmp_nle_f64,  goc_rdna4_v_cmp_neq_f64,
    goc_rdna4_v_cmp_nlt_f64,  goc_rdna4_v_cmpx_lt_f64,  goc_rdna4_v_cmpx_eq_f64,
    goc_rdna4_v_cmpx_le_f64,  goc_rdna4_v_cmpx_gt_f64,  goc_rdna4_v_cmpx_lg_f64,
    goc_rdna4_v_cmpx_ge_f64,  goc_rdna4_v_cmpx_o_f64,   goc_rdna4_v_cmpx_u_f64,
    goc_rdna4_v_cmpx_nge_f64, goc_rdna4_v_cmpx_nlg_f64, goc_rdna4_v_cmpx_ngt_f64,
    goc_rdna4_v_cmpx_nle_f64, goc_rdna4_v_cmpx_neq_f64, goc_rdna4_v_cmpx_nlt_f64,
};
static const char *const float_compare_names[] = {
    "v_cmp_lt_f16",   "v_cmp_eq_f16",   "v_cmp_le_f16",   "v_cmp_gt_f16",   "v_cmp_lg_f16",
    "v_cmp_ge_f16",   "v_cmp_o_f16",    "v_cmp_u_f16",    "v_cmp_nge_f16",  "v_cmp_nlg_f16",
    "v_cmp_ngt_f16",  "v_cmp_nle_f16",  "v_cmp_neq_f16",  "v_cmp_nlt_f16",  "v_cmpx_lt_f16",
    "v_cmpx_eq_f16",  "v_cmpx_le_f16",  "v_cmpx_gt_f16",  "v_cmpx_lg_f16",  "v_cmpx_ge_f16",
    "v_cmpx_o_f16",   "v_cmpx_u_f16",   "v_cmpx_nge_f16", "v_cmpx_nlg_f16", "v_cmpx_ngt_f16",
    "v_cmpx_nle_f16", "v_cmpx_neq_f16", "v_cmpx_nlt_f16", "v_cmp_lt_f32",   "v_cmp_eq_f32",
    "v_cmp_le_f32",   "v_cmp_gt_f32",   "v_cmp_lg_f32",   "v_cmp_ge_f32",   "v_cmp_o_f32",
    "v_cmp_u_f32",    "v_cmp_nge_f32",  "v_cmp_nlg_f32",  "v_cmp_ngt_f32",  "v_cmp_nle_f32",
    "v_cmp_neq_f32",  "v_cmp_nlt_f32",  "v_cmpx_lt_f32",  "v_cmpx_eq_f32",  "v_cmpx_le_f32",
    "v_cmpx_gt_f32",  "v_cmpx_lg_f32",  "v_cmpx_ge_f32",  "v_cmpx_o_f32",   "v_cmpx_u_f32",
    "v_cmpx_nge_f32", "v_cmpx_nlg_f32", "v_cmpx_ngt_f32", "v_cmpx_nle_f32", "v_cmpx_neq_f32",
    "v_cmpx_nlt_f32", "v_cmp_lt_f64",   "v_cmp_eq_f64",   "v_cmp_le_f64",   "v_cmp_gt_f64",
    "v_cmp_lg_f64",   "v_cmp_ge_f64",   "v_cmp_o_f64",    "v_cmp_u_f64",    "v_cmp_nge_f64",
    "v_cmp_nlg_f64",  "v_cmp_ngt_f64",  "v_cmp_nle_f64",  "v_cmp_neq_f64",  "v_cmp_nlt_f64",
    "v_cmpx_lt_f64",  "v_cmpx_eq_f64",  "v_cmpx_le_f64",  "v_cmpx_gt_f64",  "v_cmpx_lg_f64",
    "v_cmpx_ge_f64",  "v_cmpx_o_f64",   "v_cmpx_u_f64",   "v_cmpx_nge_f64", "v_cmpx_nlg_f64",
    "v_cmpx_ngt_f64", "v_cmpx_nle_f64", "v_cmpx_neq_f64", "v_cmpx_nlt_f64",
};

inline uint32_t float_compare_mode(unsigned m) {
  return (m & 1 ? GOC_ALU_ABS_A : 0) | (m & 2 ? GOC_ALU_ABS_B : 0) | (m & 4 ? GOC_ALU_NEG_A : 0) |
         (m & 8 ? GOC_ALU_NEG_B : 0) | (m & 16 ? GOC_ALU_HIGH_A : 0) |
         (m & 32 ? GOC_ALU_HIGH_B : 0);
}

inline void float_compare_inputs(unsigned fmt, unsigned i, uint32_t *w) {
  const uint16_t edges16[] = {0,      0x8000, 1,      0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400,
                              0x3c00, 0xbc00, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7c01, 0x7e01};
  const uint32_t edges32[] = {0,          0x80000000, 1,          0x80000001,
                              0x007fffff, 0x807fffff, 0x00800000, 0x80800000,
                              0x3f800000, 0xbf800000, 0x7f7fffff, 0xff7fffff,
                              0x7f800000, 0xff800000, 0x7f800001, 0x7fc00001};
  const uint64_t edges64[] = {0,
                              UINT64_C(0x8000000000000000),
                              1,
                              UINT64_C(0x8000000000000001),
                              UINT64_C(0xfffffffffffff),
                              UINT64_C(0x800fffffffffffff),
                              UINT64_C(0x10000000000000),
                              UINT64_C(0x8010000000000000),
                              UINT64_C(0x3ff0000000000000),
                              UINT64_C(0xbff0000000000000),
                              UINT64_C(0x7fefffffffffffff),
                              UINT64_C(0xffefffffffffffff),
                              UINT64_C(0x7ff0000000000000),
                              UINT64_C(0xfff0000000000000),
                              UINT64_C(0x7ff0000000000001),
                              UINT64_C(0x7ff8000000000001)};

  uint64_t a, b;
  if (i < 256) {
    a = fmt == 0 ? edges16[i % 16] : fmt == 1 ? edges32[i % 16] : edges64[i % 16];
    b = fmt == 0 ? edges16[i / 16] : fmt == 1 ? edges32[i / 16] : edges64[i / 16];
    if (fmt == 0) {
      a |= uint32_t(edges16[15 - i % 16]) << 16;
      b |= uint32_t(edges16[15 - i / 16]) << 16;
    }
  } else {
    a = (uint64_t((i * 0x7395a831u) ^ 0xa7925163u) << 32) | ((i * 0x83a1459du) ^ 0x5389d241u);
    b = (uint64_t((i * 0x38459317u) ^ 0x187abd31u) << 32) | ((i * 0x9e3779b9u) ^ 0xa5a59669u);
    if (i % 4 == 0)
      b = a;
    else if (fmt == 2 && i % 4 == 1)
      b = (a & 0xffffffff00000000ULL) | uint32_t(b);
  }
  w[0] = uint32_t(a);
  w[1] = uint32_t(a >> 32);
  w[2] = uint32_t(b);
  w[3] = uint32_t(b >> 32);
}

inline bool float_compare_reference(unsigned op, unsigned m, bool flush, const uint32_t *w) {
  unsigned fmt = op / 28, p = op % 14 + 1;
  unsigned bits = fmt == 0 ? 16 : fmt == 1 ? 32 : 64;
  uint64_t sign = UINT64_C(1) << (bits - 1), inf = fmt == 0   ? 0x7c00
                                                   : fmt == 1 ? 0x7f800000
                                                              : UINT64_C(0x7ff0000000000000);
  uint64_t a = fmt == 2   ? (uint64_t(w[1]) << 32) | w[0]
               : fmt == 0 ? ((w[0] >> (m & 16 ? 16 : 0)) & 65535)
                          : w[0];
  uint64_t b = fmt == 2   ? (uint64_t(w[3]) << 32) | w[2]
               : fmt == 0 ? ((w[2] >> (m & 32 ? 16 : 0)) & 65535)
                          : w[2];
  if (m & 1)
    a &= ~sign;
  if (m & 2)
    b &= ~sign;
  if (m & 4)
    a ^= sign;
  if (m & 8)
    b ^= sign;
  if (flush) {
    if (!(a & inf))
      a &= sign;
    if (!(b & inf))
      b &= sign;
  }
  bool nan = (a & (sign - 1)) > inf || (b & (sign - 1)) > inf;
  bool eq = a == b || (!(a & (sign - 1)) && !(b & (sign - 1)));
  bool lt = !eq && ((a ^ b) & sign ? bool(a & sign) : (a & sign ? a > b : a < b));
  bool invert = p >= 8;
  unsigned base = invert ? 15 - p : p;
  bool result = !nan && (base == 1   ? lt
                         : base == 2 ? eq
                         : base == 3 ? lt || eq
                         : base == 4 ? !lt && !eq
                         : base == 5 ? !eq
                         : base == 6 ? !lt
                                     : true);
  return result != invert;
}

} // namespace goc_test

// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"
#include "rdna4_scalar_fp_reference.h"
#include "rdna4_scalar_integer_reference.h"

#include <stdint.h>

namespace goc_test {

inline const char *const scalar_compare_names[] = {
    "s_cmp_eq_i32",  "s_cmp_lg_i32",  "s_cmp_gt_i32",  "s_cmp_ge_i32",  "s_cmp_lt_i32",
    "s_cmp_le_i32",  "s_cmp_eq_u32",  "s_cmp_lg_u32",  "s_cmp_gt_u32",  "s_cmp_ge_u32",
    "s_cmp_lt_u32",  "s_cmp_le_u32",  "s_bitcmp0_b32", "s_bitcmp1_b32", "s_bitcmp0_b64",
    "s_bitcmp1_b64", "s_cmp_eq_u64",  "s_cmp_lg_u64",  "s_cmp_lt_f32",  "s_cmp_lt_f16",
    "s_cmp_eq_f32",  "s_cmp_eq_f16",  "s_cmp_le_f32",  "s_cmp_le_f16",  "s_cmp_gt_f32",
    "s_cmp_gt_f16",  "s_cmp_lg_f32",  "s_cmp_lg_f16",  "s_cmp_ge_f32",  "s_cmp_ge_f16",
    "s_cmp_o_f32",   "s_cmp_o_f16",   "s_cmp_u_f32",   "s_cmp_u_f16",   "s_cmp_nge_f32",
    "s_cmp_nge_f16", "s_cmp_nlg_f32", "s_cmp_nlg_f16", "s_cmp_ngt_f32", "s_cmp_ngt_f16",
    "s_cmp_nle_f32", "s_cmp_nle_f16", "s_cmp_neq_f32", "s_cmp_neq_f16", "s_cmp_nlt_f32",
    "s_cmp_nlt_f16"};

inline void scalar_compare_inputs(unsigned i, unsigned op, uint32_t *w) {
  if (op < 18)
    scalar_integer_inputs(i, w);
  else {
    uint32_t f[2];
    scalar_fp_inputs(i, (op - 18) % 2, f);
    w[0] = f[0];
    w[1] = 0;
    w[2] = f[1];
    w[3] = 0;
  }
}

inline int scalar_compare_call(unsigned op, uint64_t flags, uint32_t exec_mask, uint64_t mode,
                               uint32_t *scc, uint64_t a, uint64_t b) {
  switch (op) {
  case 0:
    return goc_rdna4_s_cmp_eq_i32(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b));
  case 1:
    return goc_rdna4_s_cmp_lg_i32(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b));
  case 2:
    return goc_rdna4_s_cmp_gt_i32(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b));
  case 3:
    return goc_rdna4_s_cmp_ge_i32(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b));
  case 4:
    return goc_rdna4_s_cmp_lt_i32(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b));
  case 5:
    return goc_rdna4_s_cmp_le_i32(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b));
  case 6:
    return goc_rdna4_s_cmp_eq_u32(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b));
  case 7:
    return goc_rdna4_s_cmp_lg_u32(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b));
  case 8:
    return goc_rdna4_s_cmp_gt_u32(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b));
  case 9:
    return goc_rdna4_s_cmp_ge_u32(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b));
  case 10:
    return goc_rdna4_s_cmp_lt_u32(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b));
  case 11:
    return goc_rdna4_s_cmp_le_u32(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b));
  case 12:
    return goc_rdna4_s_bitcmp0_b32(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b));
  case 13:
    return goc_rdna4_s_bitcmp1_b32(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b));
  case 14:
    return goc_rdna4_s_bitcmp0_b64(flags, exec_mask, mode, scc, uint64_t(a), uint32_t(b));
  case 15:
    return goc_rdna4_s_bitcmp1_b64(flags, exec_mask, mode, scc, uint64_t(a), uint32_t(b));
  case 16:
    return goc_rdna4_s_cmp_eq_u64(flags, exec_mask, mode, scc, uint64_t(a), uint64_t(b));
  case 17:
    return goc_rdna4_s_cmp_lg_u64(flags, exec_mask, mode, scc, uint64_t(a), uint64_t(b));
  case 18:
    return goc_rdna4_s_cmp_lt_f32(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b), nullptr);
  case 19:
    return goc_rdna4_s_cmp_lt_f16(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b), nullptr);
  case 20:
    return goc_rdna4_s_cmp_eq_f32(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b), nullptr);
  case 21:
    return goc_rdna4_s_cmp_eq_f16(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b), nullptr);
  case 22:
    return goc_rdna4_s_cmp_le_f32(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b), nullptr);
  case 23:
    return goc_rdna4_s_cmp_le_f16(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b), nullptr);
  case 24:
    return goc_rdna4_s_cmp_gt_f32(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b), nullptr);
  case 25:
    return goc_rdna4_s_cmp_gt_f16(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b), nullptr);
  case 26:
    return goc_rdna4_s_cmp_lg_f32(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b), nullptr);
  case 27:
    return goc_rdna4_s_cmp_lg_f16(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b), nullptr);
  case 28:
    return goc_rdna4_s_cmp_ge_f32(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b), nullptr);
  case 29:
    return goc_rdna4_s_cmp_ge_f16(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b), nullptr);
  case 30:
    return goc_rdna4_s_cmp_o_f32(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b), nullptr);
  case 31:
    return goc_rdna4_s_cmp_o_f16(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b), nullptr);
  case 32:
    return goc_rdna4_s_cmp_u_f32(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b), nullptr);
  case 33:
    return goc_rdna4_s_cmp_u_f16(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b), nullptr);
  case 34:
    return goc_rdna4_s_cmp_nge_f32(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b), nullptr);
  case 35:
    return goc_rdna4_s_cmp_nge_f16(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b), nullptr);
  case 36:
    return goc_rdna4_s_cmp_nlg_f32(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b), nullptr);
  case 37:
    return goc_rdna4_s_cmp_nlg_f16(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b), nullptr);
  case 38:
    return goc_rdna4_s_cmp_ngt_f32(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b), nullptr);
  case 39:
    return goc_rdna4_s_cmp_ngt_f16(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b), nullptr);
  case 40:
    return goc_rdna4_s_cmp_nle_f32(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b), nullptr);
  case 41:
    return goc_rdna4_s_cmp_nle_f16(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b), nullptr);
  case 42:
    return goc_rdna4_s_cmp_neq_f32(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b), nullptr);
  case 43:
    return goc_rdna4_s_cmp_neq_f16(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b), nullptr);
  case 44:
    return goc_rdna4_s_cmp_nlt_f32(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b), nullptr);
  case 45:
    return goc_rdna4_s_cmp_nlt_f16(flags, exec_mask, mode, scc, uint32_t(a), uint32_t(b), nullptr);
  }
  return GOC_ERROR_INVALID_FLAGS;
}

} // namespace goc_test

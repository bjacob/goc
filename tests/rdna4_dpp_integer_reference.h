// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"
#include "rdna4_bit_count_reference.h"
#include "rdna4_shift_reference.h"

#include <stdint.h>

namespace goc_test {

inline const char *const dpp_integer_names[] = {
    "v_clz_i32_u32",      "v_ctz_i32_b32", "v_cls_i32",     "v_bcnt_u32_b32", "v_mbcnt_lo_u32_b32",
    "v_mbcnt_hi_u32_b32", "v_and_b32",     "v_or_b32",      "v_xor_b32",      "v_xnor_b32",
    "v_not_b32",          "v_lshlrev_b32", "v_lshrrev_b32", "v_ashrrev_i32"};

inline int dpp_not(uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a, const uint32_t *const *) {
  return goc_rdna4_v_not_b32(flags, mask, mode, d, a);
}

using DppIntegerFn = decltype(&goc_rdna4_v_bcnt_u32_b32);
inline const DppIntegerFn dpp_integer_functions[] = {count_leading,
                                                     count_trailing,
                                                     count_sign,
                                                     goc_rdna4_v_bcnt_u32_b32,
                                                     goc_rdna4_v_mbcnt_lo_u32_b32,
                                                     goc_rdna4_v_mbcnt_hi_u32_b32,
                                                     goc_rdna4_v_and_b32,
                                                     goc_rdna4_v_or_b32,
                                                     goc_rdna4_v_xor_b32,
                                                     goc_rdna4_v_xnor_b32,
                                                     dpp_not,
                                                     goc_rdna4_v_lshlrev_b32,
                                                     goc_rdna4_v_lshrrev_b32,
                                                     goc_rdna4_v_ashrrev_i32};

inline uint32_t dpp_integer_reference(unsigned op, uint32_t a, uint32_t b, unsigned lane) {
  if (op < 6)
    return bit_count_reference(op, a, b, lane);
  if (op >= 11)
    return uint32_t(shift_reference(32, op - 11, a, b));
  switch (op) {
  case 6:
    return a & b;
  case 7:
    return a | b;
  case 8:
    return a ^ b;
  case 9:
    return ~(a ^ b);
  default:
    return ~a;
  }
}

} // namespace goc_test

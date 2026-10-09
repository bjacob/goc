// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"
#include "rdna4_scalar_integer_reference.h"

#include <stdint.h>

namespace goc_test {

inline const char *const scalar_field_names[] = {
    "s_bfe_u32",     "s_bfe_i32",       "s_bfe_u64",       "s_bfe_i64",       "s_bfm_b32",
    "s_bfm_b64",     "s_bcnt0_i32_b32", "s_bcnt0_i32_b64", "s_bcnt1_i32_b32", "s_bcnt1_i32_b64",
    "s_ctz_i32_b32", "s_ctz_i32_b64",   "s_clz_i32_u32",   "s_clz_i32_u64",   "s_cls_i32",
    "s_cls_i32_i64", "s_bitset0_b32",   "s_bitset0_b64",   "s_bitset1_b32",   "s_bitset1_b64"};

inline void scalar_field_inputs(unsigned i, uint32_t *w) {
  scalar_integer_inputs(i, w);
  if (i < 8192) {
    const uint64_t edges[] = {0,
                              1,
                              2,
                              0x7fffffffULL,
                              0x80000000ULL,
                              0xffffffffULL,
                              0x100000000ULL,
                              0x8000000000000000ULL,
                              0x7fffffffffffffffULL,
                              0xffffffffffffffffULL,
                              0x5555555555555555ULL,
                              0xaaaaaaaaaaaaaaaaULL,
                              0x123456789abcdef0ULL,
                              0xfedcba9876543210ULL,
                              0x0000ffff0000ffffULL,
                              0xffff0000ffff0000ULL};
    uint64_t a = edges[(i / 64) % 16];
    w[0] = uint32_t(a);
    w[1] = uint32_t(a >> 32);
    w[2] = ((i % 128) << 16) | ((i / 128) % 64) | 0xa580ffc0u;
  }
}

inline int scalar_field_call(unsigned op, uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *d,
                             uint64_t *d64, uint64_t a, uint32_t b, uint32_t *scc) {
  switch (op) {
  case 0:
    return goc_rdna4_s_bfe_u32(flags, mask, mode, d, uint32_t(a), b, scc);
  case 1:
    return goc_rdna4_s_bfe_i32(flags, mask, mode, d, uint32_t(a), b, scc);
  case 2:
    return goc_rdna4_s_bfe_u64(flags, mask, mode, d64, uint64_t(a), b, scc);
  case 3:
    return goc_rdna4_s_bfe_i64(flags, mask, mode, d64, uint64_t(a), b, scc);
  case 4:
    return goc_rdna4_s_bfm_b32(flags, mask, mode, d, uint32_t(a), b);
  case 5:
    return goc_rdna4_s_bfm_b64(flags, mask, mode, d64, uint32_t(a), b);
  case 6:
    return goc_rdna4_s_bcnt0_i32_b32(flags, mask, mode, d, uint32_t(a), scc);
  case 7:
    return goc_rdna4_s_bcnt0_i32_b64(flags, mask, mode, d, uint64_t(a), scc);
  case 8:
    return goc_rdna4_s_bcnt1_i32_b32(flags, mask, mode, d, uint32_t(a), scc);
  case 9:
    return goc_rdna4_s_bcnt1_i32_b64(flags, mask, mode, d, uint64_t(a), scc);
  case 10:
    return goc_rdna4_s_ctz_i32_b32(flags, mask, mode, d, uint32_t(a));
  case 11:
    return goc_rdna4_s_ctz_i32_b64(flags, mask, mode, d, uint64_t(a));
  case 12:
    return goc_rdna4_s_clz_i32_u32(flags, mask, mode, d, uint32_t(a));
  case 13:
    return goc_rdna4_s_clz_i32_u64(flags, mask, mode, d, uint64_t(a));
  case 14:
    return goc_rdna4_s_cls_i32(flags, mask, mode, d, uint32_t(a));
  case 15:
    return goc_rdna4_s_cls_i32_i64(flags, mask, mode, d, uint64_t(a));
  case 16:
    return goc_rdna4_s_bitset0_b32(flags, mask, mode, d, b);
  case 17:
    return goc_rdna4_s_bitset0_b64(flags, mask, mode, d64, b);
  case 18:
    return goc_rdna4_s_bitset1_b32(flags, mask, mode, d, b);
  case 19:
    return goc_rdna4_s_bitset1_b64(flags, mask, mode, d64, b);
  }
  return GOC_ERROR_INVALID_FLAGS;
}

} // namespace goc_test

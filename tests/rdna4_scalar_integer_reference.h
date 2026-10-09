// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <stdint.h>

namespace goc_test {

inline const char *const scalar_integer_names[] = {
    "s_add_co_u32",    "s_sub_co_u32", "s_add_co_i32",  "s_sub_co_i32",  "s_add_co_ci_u32",
    "s_sub_co_ci_u32", "s_abs_i32",    "s_absdiff_i32", "s_min_i32",     "s_min_u32",
    "s_max_i32",       "s_max_u32",    "s_mul_i32",     "s_mul_hi_u32",  "s_mul_hi_i32",
    "s_add_nc_u64",    "s_sub_nc_u64", "s_mul_u64",     "s_addk_co_i32", "s_mulk_i32"};

inline const int scalar_integer_literals[] = {0,      1, -1, 32767,  -32768, 32766,
                                              -32767, 2, -2, 0x1234, -0x1234};

inline void scalar_integer_inputs(unsigned i, uint32_t *w) {
  const uint64_t edges[] = {0,
                            1,
                            2,
                            0x7ffffffeULL,
                            0x7fffffffULL,
                            0x80000000ULL,
                            0x80000001ULL,
                            0xfffffffeULL,
                            0xffffffffULL,
                            0x100000000ULL,
                            0x7ffffffffffffffeULL,
                            0x7fffffffffffffffULL,
                            0x8000000000000000ULL,
                            0x8000000000000001ULL,
                            0xfffffffffffffffeULL,
                            0xffffffffffffffffULL};
  uint64_t a, b;
  if (i < 256) {
    a = edges[i % 16];
    b = edges[i / 16];
  } else {
    a = (uint64_t((i * 0x7395a831u) ^ 0xa7925163u) << 32) | ((i * 0x83a1459du) ^ 0x5389d241u);
    b = (uint64_t((i * 0x38459317u) ^ 0x187abd31u) << 32) | ((i * 0x9e3779b9u) ^ 0xa5a59669u);
    if (i % 4 == 0)
      b = a;
  }
  w[0] = uint32_t(a);
  w[1] = uint32_t(a >> 32);
  w[2] = uint32_t(b);
  w[3] = uint32_t(b >> 32);
}

inline int scalar_integer_call(unsigned op, uint64_t flags, uint64_t mask, uint32_t mode,
                               uint32_t *d, uint64_t *d64, uint64_t a, uint64_t b, uint32_t *scc,
                               uint32_t seed, uint16_t immediate) {
  switch (op) {
  case 0:
    return goc_rdna4_s_add_co_u32(flags, mask, mode, d, uint32_t(a), uint32_t(b), scc);
  case 1:
    return goc_rdna4_s_sub_co_u32(flags, mask, mode, d, uint32_t(a), uint32_t(b), scc);
  case 2:
    return goc_rdna4_s_add_co_i32(flags, mask, mode, d, uint32_t(a), uint32_t(b), scc);
  case 3:
    return goc_rdna4_s_sub_co_i32(flags, mask, mode, d, uint32_t(a), uint32_t(b), scc);
  case 4:
    return goc_rdna4_s_add_co_ci_u32(flags, mask, mode, d, uint32_t(a), uint32_t(b), scc, seed);
  case 5:
    return goc_rdna4_s_sub_co_ci_u32(flags, mask, mode, d, uint32_t(a), uint32_t(b), scc, seed);
  case 6:
    return goc_rdna4_s_abs_i32(flags, mask, mode, d, uint32_t(a), scc);
  case 7:
    return goc_rdna4_s_absdiff_i32(flags, mask, mode, d, uint32_t(a), uint32_t(b), scc);
  case 8:
    return goc_rdna4_s_min_i32(flags, mask, mode, d, uint32_t(a), uint32_t(b), scc);
  case 9:
    return goc_rdna4_s_min_u32(flags, mask, mode, d, uint32_t(a), uint32_t(b), scc);
  case 10:
    return goc_rdna4_s_max_i32(flags, mask, mode, d, uint32_t(a), uint32_t(b), scc);
  case 11:
    return goc_rdna4_s_max_u32(flags, mask, mode, d, uint32_t(a), uint32_t(b), scc);
  case 12:
    return goc_rdna4_s_mul_i32(flags, mask, mode, d, uint32_t(a), uint32_t(b));
  case 13:
    return goc_rdna4_s_mul_hi_u32(flags, mask, mode, d, uint32_t(a), uint32_t(b));
  case 14:
    return goc_rdna4_s_mul_hi_i32(flags, mask, mode, d, uint32_t(a), uint32_t(b));
  case 15:
    return goc_rdna4_s_add_nc_u64(flags, mask, mode, d64, a, b);
  case 16:
    return goc_rdna4_s_sub_nc_u64(flags, mask, mode, d64, a, b);
  case 17:
    return goc_rdna4_s_mul_u64(flags, mask, mode, d64, a, b);
  case 18:
    return goc_rdna4_s_addk_co_i32(flags, mask, mode, d, immediate, scc);
  case 19:
    return goc_rdna4_s_mulk_i32(flags, mask, mode, d, immediate);
  }
  return GOC_ERROR_INVALID_FLAGS;
}

} // namespace goc_test

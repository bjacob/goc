// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <stdint.h>

namespace goc_test {

using ScalarFpFn = decltype(&goc_rdna4_s_add_f32);
inline const ScalarFpFn scalar_fp_functions[] = {
    goc_rdna4_s_add_f32,     goc_rdna4_s_add_f16,     goc_rdna4_s_sub_f32,
    goc_rdna4_s_sub_f16,     goc_rdna4_s_mul_f32,     goc_rdna4_s_mul_f16,
    goc_rdna4_s_min_num_f32, goc_rdna4_s_min_num_f16, goc_rdna4_s_max_num_f32,
    goc_rdna4_s_max_num_f16, goc_rdna4_s_minimum_f32, goc_rdna4_s_minimum_f16,
    goc_rdna4_s_maximum_f32, goc_rdna4_s_maximum_f16};

inline const char *const scalar_fp_names[] = {
    "s_add_f32",     "s_add_f16",     "s_sub_f32",     "s_sub_f16",     "s_mul_f32",
    "s_mul_f16",     "s_min_num_f32", "s_min_num_f16", "s_max_num_f32", "s_max_num_f16",
    "s_minimum_f32", "s_minimum_f16", "s_maximum_f32", "s_maximum_f16"};

inline void scalar_fp_inputs(unsigned i, bool half, uint32_t *w) {
  const uint32_t e32[] = {0,          0x80000000, 1,          0x80000001, 0x007fffff, 0x807fffff,
                          0x00800000, 0x80800000, 0x3f800000, 0xbf800000, 0x3f000000, 0xbf000000,
                          0x40000000, 0xc0000000, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
                          0x7fc00000, 0xffc00001, 0x7f800001, 0xff800001, 0x3f800001, 0xbf800001,
                          0x3f7fffff, 0x00800001, 0x33800000, 0xb3800000, 0x477fe000, 0xc77fe000,
                          0x38800000, 0x33000000};
  const uint32_t e16[] = {0,      0x8000, 1,      0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400,
                          0x3c00, 0xbc00, 0x3800, 0xb800, 0x4000, 0xc000, 0x7bff, 0xfbff,
                          0x7c00, 0xfc00, 0x7e00, 0xfe01, 0x7c01, 0xfc01, 0x3c01, 0xbc01,
                          0x3bff, 0x0401, 0x1000, 0x9000, 0x7bfe, 0xfbfe, 0x0800, 0x1400};
  for (unsigned j = 0; j < 2; ++j) {
    uint32_t v = i < 1024
                     ? (half ? e16 : e32)[j ? i / 32 : i % 32]
                     : ((i * (j ? 0x9e3779b9u : 0x7395a831u)) ^ (j ? 0xa5a59669u : 0xa7925163u));
    w[j] = half ? ((v & 65535) | 0xabcd0000u) : v;
  }
}

inline void scalar_fp_boundary_inputs(unsigned i, bool half, uint32_t *w) {
  w[0] = (half ? 0x3c00u : 0x3f800000u) + (i % 64) - 32;
  w[1] = (half ? 0x400u : 0x800000u) + (i / 64) - 32;
}

inline uint64_t scalar_fp_flags(unsigned state) {
  return (state & 4 ? GOC_FP16_OVFL : 0) | (state & 1 ? 0 : GOC_FP_FLUSH_INPUT_DENORMALS) |
         (state & 2 ? 0 : GOC_FP_FLUSH_OUTPUT_DENORMALS);
}

// NaN signs/payloads are unspecified; all other bits must match in these tests.
inline uint32_t scalar_fp_canonical(uint32_t v, bool half) {
  uint32_t inf = half ? 0x7c00 : 0x7f800000, mag = half ? 0x7fff : 0x7fffffff;
  return ((v & mag) > inf && (!half || !(v >> 16))) ? inf + 1 : v;
}

} // namespace goc_test

// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <stdint.h>

namespace goc_test {

using PseudoScalarFn = decltype(&goc_rdna4_v_s_exp_f32);
static const PseudoScalarFn pseudo_scalar_functions[] = {
    goc_rdna4_v_s_exp_f32,  goc_rdna4_v_s_exp_f16,  goc_rdna4_v_s_log_f32, goc_rdna4_v_s_log_f16,
    goc_rdna4_v_s_rcp_f32,  goc_rdna4_v_s_rcp_f16,  goc_rdna4_v_s_rsq_f32, goc_rdna4_v_s_rsq_f16,
    goc_rdna4_v_s_sqrt_f32, goc_rdna4_v_s_sqrt_f16,
};
static const char *const pseudo_scalar_names[] = {
    "v_s_exp_f32", "v_s_exp_f16", "v_s_log_f32", "v_s_log_f16",  "v_s_rcp_f32",
    "v_s_rcp_f16", "v_s_rsq_f32", "v_s_rsq_f16", "v_s_sqrt_f32", "v_s_sqrt_f16",
};

inline uint32_t pseudo_scalar_mode(unsigned m) {
  return (m & 1 ? GOC_ALU_ABS_A : 0) | (m & 2 ? GOC_ALU_NEG_A : 0) | (((m >> 2) & 3) << 6) |
         (m & 16 ? GOC_ALU_CLAMP : 0);
}

inline uint64_t pseudo_scalar_flags(unsigned state) {
  return (state & 4 ? GOC_FP16_OVFL : 0) | (state & 1 ? 0 : GOC_FP_FLUSH_INPUT_DENORMALS) |
         (state & 2 ? 0 : GOC_FP_FLUSH_OUTPUT_DENORMALS);
}

inline uint32_t pseudo_scalar_input(unsigned half, unsigned i) {
  const uint32_t edges32[] = {
      0,          0x80000000, 1,          0x80000001, 0x007fffff, 0x807fffff, 0x00800000,
      0x80800000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x7f7fffff, 0xff7fffff,
      0x7f800000, 0xff800000, 0x7f800001, 0x7fc00001, 0xff800001, 0xffc00001, 0x42ff0000,
      0xc3160000, 0x3f000000, 0xbf000000, 0x3eaaaaab, 0x40400000, 0x40800000, 0x3f800001,
      0x3f7fffff, 0x41000000, 0x42000000, 0x43000000};
  const uint16_t edges16[] = {0,      0x8000, 1,      0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400,
                              0x3c00, 0xbc00, 0x4000, 0xc000, 0x7bff, 0xfbff, 0x7c00, 0xfc00,
                              0x7c01, 0x7e01, 0xfc01, 0xfe01, 0x4bff, 0xcc40, 0x3800, 0xb800,
                              0x3555, 0x4200, 0x4400, 0x3c01, 0x3bff, 0x4800, 0x5000, 0x5800};

  return half ? (0xabcd0000u | (i < 32 ? edges16[i] : uint16_t(i * 17)))
              : (i < 32 ? edges32[i] : ((i * 0x7395a831u) ^ 0xa7925163u));
}

// Finite nonzero results allow one half ULP or two single ULPs. Signed zeros and
// infinities must match exactly; NaN payloads and signs are unspecified.
inline bool pseudo_scalar_close(uint32_t a, uint32_t b, bool half) {
  uint32_t inf = half ? 0x7c00 : 0x7f800000, mag = half ? 0x7fff : 0x7fffffff;
  if (half && (a >> 16))
    return false;
  if ((a & mag) > inf)
    return (b & mag) > inf;
  if ((a & mag) == inf || (b & mag) >= inf || !(a & mag) || !(b & mag))
    return a == b;
  return a > b ? a - b <= unsigned(half ? 1 : 2) : b - a <= unsigned(half ? 1 : 2);
}

} // namespace goc_test

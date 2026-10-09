// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"
#include "rdna4_scalar_fp_reference.h"

#include <stdint.h>

namespace goc_test {

inline const char *const scalar_fma_names[] = {"s_fmac_f32", "s_fmac_f16", "s_fmaak_f32",
                                               "s_fmamk_f32"};

inline const uint32_t scalar_fma_literals[] = {0,          0x80000000, 1,          0x80000001,
                                               0x3f800000, 0xbf800000, 0x3f800001, 0x7f800000,
                                               0xff800000, 0x7fc00000, 0x7f800001, 0x00800000};

inline void scalar_fma_inputs(unsigned i, bool half, uint32_t *w) {
  scalar_fp_inputs(i, half, w);
  uint32_t temp[2];
  scalar_fp_inputs((i * 13 + 17) % 4096, half, temp);
  w[2] = temp[0];
  // Cancellation, fused rounding, and minimum-normal underflow boundaries.
  if (i < 64) {
    w[0] = half ? 0x3c01u : 0x3f800001u;
    w[1] = half ? 0x3bfeu : 0x3f7ffffeu;
    w[2] = half ? 0xbc00u : 0xbf800000u;
    if (i & 1)
      w[0] ^= half ? 0x8000u : 0x80000000u;
    if (i & 2)
      w[2] ^= half ? 0x8000u : 0x80000000u;
    if (i & 4) {
      w[0] = half ? 0x400u : 0x800000u;
      w[1] = (half ? 0x3c00u : 0x3f800000u) - (i % 4);
      w[2] = (i & 8) ? (half ? 0x8000u : 0x80000000u) : 0;
    }
    if (i & 16) {
      w[0] = half ? 1 : 1;
      w[1] = half ? 0x3800u : 0x3f000000u;
      w[2] = (half ? 0x400u : 0x800000u) - (i % 4);
    }
    if (i & 32)
      w[2] ^= half ? 0x8000u : 0x80000000u;
  }
}

// Variant 0/1 is FMAC; variants 2..13 / 14..25 are FMAAK / FMAMK literals.
inline int scalar_fma_call(unsigned variant, uint64_t flags, uint64_t mode, uint32_t *d, uint32_t a,
                           uint32_t b) {
  if (variant == 0)
    return goc_rdna4_s_fmac_f32(flags, mode, d, a, b, nullptr);
  if (variant == 1)
    return goc_rdna4_s_fmac_f16(flags, mode, d, a, b, nullptr);
  if (variant < 14)
    return goc_rdna4_s_fmaak_f32(flags, mode, d, a, b, scalar_fma_literals[variant - 2], nullptr);
  return goc_rdna4_s_fmamk_f32(flags, mode, d, a, scalar_fma_literals[variant - 14], b, nullptr);
}

} // namespace goc_test

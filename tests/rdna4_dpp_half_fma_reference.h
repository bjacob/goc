// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"
#include "rdna4_half_fma_reference.h"

#include <stdint.h>

namespace goc_test {

inline int half_fmac(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *) {
  return goc_rdna4_v_fmac_f16(flags, mask, mode, d, a, b);
}

using HalfFmaFn = decltype(&goc_rdna4_v_fma_f16);
const HalfFmaFn half_fma_functions[] = {goc_rdna4_v_fma_f16, half_fmac};
const char *const half_fma_names[] = {"v_fma_f16", "v_fmac_f16"};

inline bool half_fma_valid_mode(unsigned op, uint32_t mode) {
  return op == 0 || !(mode & (GOC_ALU_NEG_C | GOC_ALU_ABS_C | GOC_ALU_HIGH_C));
}

inline uint32_t half_fma_result(unsigned op, uint32_t a, uint32_t b, uint32_t c, uint32_t mode,
                                uint32_t old_d, bool saturate = false) {
  if (op == 1) {
    c = old_d;
    if (mode & GOC_ALU_HIGH_D)
      mode |= GOC_ALU_HIGH_C;
  }
  uint32_t value = half_fma_reference::evaluate(a, b, c, mode, saturate);
  unsigned shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  return (old_d & ~(65535U << shift)) | (value << shift);
}

} // namespace goc_test

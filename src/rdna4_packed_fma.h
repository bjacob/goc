// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <stdint.h>

namespace goc {

// Return ordinary ALU source and clamp flags for one packed result half.
inline uint32_t packed_half_mode(uint32_t mode, bool high) {
  // Three selector bits name A/B/C. The high-result flags invert the default
  // high-half choice, whereas low-result flags enable high-half selection.
  uint32_t selected = high ? (~(mode >> 10) & 7) : ((mode >> 7) & 7);
  uint32_t negation = (mode >> (high ? 3 : 0)) & 7;
  return negation | (selected << 9) | ((mode & GOC_PK_CLAMP) ? GOC_ALU_CLAMP : 0);
}

void packed_fma_x86_64_v3(bool saturate, uint32_t mask, uint32_t mode, uint32_t *d,
                          const uint32_t *a, const uint32_t *b, const uint32_t *c);

} // namespace goc

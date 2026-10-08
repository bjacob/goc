// SPDX-License-Identifier: MIT

#include "goc/detail/goc_rdna4.h"

// Repeated inclusion must be harmless.
#include "goc/detail/goc_rdna4.h"

#include <stdint.h>

int main(void) {
  uint32_t words[32] = {0};
  uint32_t *vgpr = words;
  const uint32_t *input = words;
  return goc_rdna4_v_fma_f32(0, 1, 0, &vgpr, &input, &input, &input);
}

// SPDX-License-Identifier: MIT

#include "goc_rdna4.h"

// Repeated inclusion must be harmless.
#include "goc_rdna4.h"

#include <stdint.h>

int main(void) {
  uint32_t words[32] = {0};
  uint32_t *vgpr = words;
  return goc_rdna4_v_fma_f32(0, 1, 0, &vgpr, &vgpr, &vgpr, &vgpr);
}

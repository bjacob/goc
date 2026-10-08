// SPDX-License-Identifier: MIT

// Deliberately test that the umbrella exposes both public components.
#include "@PROJECT_SOURCE_DIR@/include/goc/goc.h"

// Repeated inclusion must be harmless.
#include "@PROJECT_SOURCE_DIR@/include/goc/goc.h"

#include <stdint.h>

int main(void) {
  uint32_t words[32] = {0};
  uint32_t *vgpr = words;
  const uint32_t *input = words;
  return goc_rdna4_v_fma_f32(goc_init_cpu_flags(), 1, 0, &vgpr, &input, &input, &input) !=
         GOC_SUCCESS;
}

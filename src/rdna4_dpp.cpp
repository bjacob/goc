// SPDX-License-Identifier: MIT

// Inactive-source behavior follows rocjitsu's shared DPP8 execution plan.

#include "rdna4_dpp.h"
#include "goc/goc.h"

#include <stdint.h>

namespace goc {

void dpp8_source(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *out, const uint32_t *a) {
  uint32_t selectors = uint32_t(mode >> GOC_DPP8_SELECT_SHIFT);
  bool fi = (mode & GOC_DPP_FI) != 0;
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    dpp8_x86_64_v4(mask, selectors, fi, out, a);
    return;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    dpp8_x86_64_v3(mask, selectors, fi, out, a);
    return;
  }
#endif
  (void)flags;
  for (unsigned lane = 0; lane < 32; ++lane) {
    unsigned src = (lane & ~7u) | ((selectors >> (3 * (lane & 7))) & 7);
    out[lane] = (fi || ((mask >> src) & 1)) ? a[src] : 0;
  }
}

} // namespace goc

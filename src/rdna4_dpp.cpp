// SPDX-License-Identifier: MIT

// Inactive-source behavior follows rocjitsu's shared DPP execution plans.

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

bool valid_dpp(uint64_t mode) {
  uint64_t high = mode & ~UINT64_C(0xffffffff);
  if (!high)
    return true;
  if (mode & GOC_DPP8)
    return !(high & ~(GOC_DPP8 | GOC_DPP_FI | GOC_DPP8_SELECT_MASK));
  if (!(mode & GOC_DPP16) || (high & ~(GOC_DPP16 | GOC_DPP_FI | GOC_DPP_BOUND_CTRL |
                                       GOC_DPP_CTRL_MASK | GOC_DPP_ROW_MASK | GOC_DPP_BANK_MASK)))
    return false;
  unsigned ctrl = unsigned((mode & GOC_DPP_CTRL_MASK) >> GOC_DPP_CTRL_SHIFT);
  return ctrl <= 0xff || (ctrl >= 0x101 && ctrl <= 0x10f) || (ctrl >= 0x111 && ctrl <= 0x11f) ||
         (ctrl >= 0x121 && ctrl <= 0x12f) || ctrl == 0x140 || ctrl == 0x141 ||
         (ctrl >= 0x150 && ctrl <= 0x16f);
}

uint32_t dpp16_source(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *out,
                      const uint32_t *a) {
  unsigned ctrl = unsigned((mode & GOC_DPP_CTRL_MASK) >> GOC_DPP_CTRL_SHIFT);
  unsigned valid_row = 0xffff;
  if (ctrl >= 0x101 && ctrl <= 0x10f)
    valid_row >>= ctrl & 15;
  if (ctrl >= 0x111 && ctrl <= 0x11f)
    valid_row = (valid_row << (ctrl & 15)) & 0xffff;
  bool fi = (mode & GOC_DPP_FI) != 0;
  uint32_t readable = 0;
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4)
    readable = dpp16_x86_64_v4(mask, ctrl, valid_row, fi, out, a);
  else
#endif
#if defined(GOC_HAVE_X86_64_V3)
      if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3)
    readable = dpp16_x86_64_v3(mask, ctrl, valid_row, fi, out, a);
  else
#endif
  {
    for (unsigned lane = 0; lane < 32; ++lane) {
      unsigned sub = lane & 15, index = sub;
      if (ctrl <= 0xff)
        index = (sub & ~3u) | ((ctrl >> (2 * (sub & 3))) & 3);
      else if (ctrl <= 0x10f)
        index = (sub + (ctrl & 15)) & 15;
      else if (ctrl <= 0x12f)
        index = (sub - (ctrl & 15)) & 15;
      else if (ctrl == 0x140)
        index = sub ^ 15;
      else if (ctrl == 0x141)
        index = sub ^ 7;
      else if (ctrl <= 0x15f)
        index = ctrl & 15;
      else
        index = sub ^ (ctrl & 15);
      unsigned src = (lane & 16) | index;
      bool read = ((valid_row >> sub) & 1) && (fi || ((mask >> src) & 1));
      out[lane] = read ? a[src] : 0;
      readable |= uint32_t(read) << lane;
    }
  }
  (void)flags;
  unsigned rows = unsigned((mode & GOC_DPP_ROW_MASK) >> GOC_DPP_ROW_SHIFT);
  unsigned banks = unsigned((mode & GOC_DPP_BANK_MASK) >> GOC_DPP_BANK_SHIFT);
  uint32_t bank_lanes = (banks & 1) * 15 | ((banks >> 1) & 1) * 0xf0 | ((banks >> 2) & 1) * 0xf00 |
                        ((banks >> 3) & 1) * 0xf000;
  uint32_t destinations = (bank_lanes * (rows & 1)) | ((bank_lanes * ((rows >> 1) & 1)) << 16);
  return mask & destinations & (mode & GOC_DPP_BOUND_CTRL ? UINT32_MAX : readable);
}

} // namespace goc

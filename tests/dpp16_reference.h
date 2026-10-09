// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"
#include <stdint.h>

namespace goc_test {

inline bool dpp16_control(unsigned ctrl) {
  return ctrl < 256 || (ctrl > 0x100 && ctrl < 0x110) || (ctrl > 0x110 && ctrl < 0x120) ||
         (ctrl > 0x120 && ctrl < 0x130) || ctrl == 0x140 || ctrl == 0x141 ||
         (ctrl >= 0x150 && ctrl < 0x170);
}

inline uint64_t dpp16_mode(unsigned ctrl, unsigned fi, unsigned bound, unsigned rows,
                           unsigned banks) {
  return GOC_DPP16 | (fi ? GOC_DPP_FI : 0) | (bound ? GOC_DPP_BOUND_CTRL : 0) |
         (uint64_t(ctrl) << GOC_DPP_CTRL_SHIFT) | (uint64_t(rows) << GOC_DPP_ROW_SHIFT) |
         (uint64_t(banks) << GOC_DPP_BANK_SHIFT);
}

// Returns whether the destination is written; source is -1 for a zero input.
inline bool dpp16_reference(unsigned ctrl, bool fi, bool bound, unsigned rows, unsigned banks,
                            uint32_t exec_mask, unsigned lane, int &source) {
  int sub = int(lane % 16), selected = sub;
  if (ctrl < 256)
    selected = (sub / 4) * 4 + int((ctrl >> (2 * (sub % 4))) & 3);
  else if (ctrl < 0x110)
    selected = sub + int(ctrl - 0x100);
  else if (ctrl < 0x120)
    selected = sub - int(ctrl - 0x110);
  else if (ctrl < 0x130)
    selected = (sub - int(ctrl - 0x120) + 16) % 16;
  else if (ctrl == 0x140)
    selected = 15 - sub;
  else if (ctrl == 0x141)
    selected = (sub / 8) * 8 + 7 - sub % 8;
  else if (ctrl < 0x160)
    selected = int(ctrl - 0x150);
  else
    selected = sub ^ int(ctrl - 0x160);
  source = selected < 0 || selected >= 16 ? -1 : int(lane / 16) * 16 + selected;
  if (source >= 0 && !fi && !(exec_mask & (1u << source)))
    source = -1;
  return (exec_mask & (1u << lane)) && (rows & (1u << (lane / 16))) &&
         (banks & (1u << ((lane % 16) / 4))) && (source >= 0 || bound);
}

} // namespace goc_test

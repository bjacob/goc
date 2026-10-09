// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"
#include "rdna4_dpp16_reference.h"

#include <stdint.h>

namespace goc_test {

// DPP8 reverse with FI off/on; DPP16 row shift with all FI/BOUND pairs;
// and row XOR with restricted row/bank enables.
inline const uint64_t dpp_modes[] = {
    0x539770100000000ULL, 0x539770300000000ULL, 0x1ff010400000000ULL, 0x1ff010c00000000ULL,
    0x1ff010600000000ULL, 0x1ff010e00000000ULL, 0xa3670c00000000ULL};

inline bool dpp_source(uint64_t mode, uint32_t mask, unsigned lane, int &source) {
  if (mode & GOC_DPP8) {
    unsigned selectors = unsigned(mode >> GOC_DPP8_SELECT_SHIFT);
    source = int(lane / 8) * 8 + int((selectors >> (3 * (lane % 8))) & 7);
    if (!(mode & GOC_DPP_FI) && !(mask & (1u << source)))
      source = -1;
    return (mask & (1u << lane)) != 0;
  }
  return dpp16_reference(
      unsigned((mode & GOC_DPP_CTRL_MASK) >> GOC_DPP_CTRL_SHIFT), mode & GOC_DPP_FI,
      mode & GOC_DPP_BOUND_CTRL, unsigned((mode & GOC_DPP_ROW_MASK) >> GOC_DPP_ROW_SHIFT),
      unsigned((mode & GOC_DPP_BANK_MASK) >> GOC_DPP_BANK_SHIFT), mask, lane, source);
}

} // namespace goc_test

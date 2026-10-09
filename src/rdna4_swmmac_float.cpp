// SPDX-License-Identifier: MIT

#include "rdna4_swmmac_float.h"
#include "goc/goc.h"
#include "internal.h"

#include <cmath>
#include <stdint.h>

namespace goc {

void swmmac_float_accumulate(uint32_t mode, SwmmacFloatInputs &input) {
  const uint32_t flip[] = {mode & GOC_WMMA_NEG_LO_B ? 0x80000000u : 0,
                           mode & GOC_WMMA_NEG_HI_B ? 0x80000000u : 0};
  for (unsigned row = 0; row < 16; ++row)
    for (unsigned col = 0; col < 16; ++col) {
      float acc = input.acc[row][col];
      for (unsigned ck = 0; ck < 16; ++ck) {
        float bv =
            goc::as_float(goc::as_bits(input.b[input.selected[row][ck]][col]) ^ flip[ck & 1]);
        acc = std::fma(input.a[row][ck], bv, acc);
      }
      input.acc[row][col] = acc;
    }
}

} // namespace goc

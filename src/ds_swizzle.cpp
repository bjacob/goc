// SPDX-License-Identifier: MIT

// Basic swizzles follow rocjitsu. Rotate/FFT follow the common ISA pseudocode,
// checked against RX 9070 captures; RDNA3's masked-rotate examples disagree
// with its own pseudocode. Rotations wrap within each 32-lane row.

#include "goc/goc.h"
#include "internal.h"

#include <stdint.h>

namespace {

template <unsigned Lanes>
int swizzle(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
            const uint32_t *const *a, uint16_t offset) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, uint32_t(instruction_flags), true))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
  unsigned mask = offset & 31, mask_bits = 0;
  for (unsigned bit = 0; bit < 5; ++bit)
    mask_bits += (mask >> bit) & 1;
  uint32_t result[Lanes];
  for (unsigned lane = 0; lane < Lanes; ++lane) {
    unsigned source;
    if (offset >= 0xe000) {
      unsigned reversed = ((lane & 1) << 4) | ((lane & 2) << 2) | (lane & 4) | ((lane & 8) >> 2) |
                          ((lane & 16) >> 4);
      source = (reversed >> mask_bits) | (lane & mask);
    } else if (offset >= 0xc000) {
      unsigned rotate = (offset >> 5) & 31;
      if (offset & 1024)
        rotate = 0U - rotate;
      source = (lane & mask) | ((lane + rotate) & ~mask);
    } else if (offset & 0x8000) {
      source = (lane & ~3U) | ((offset >> (2 * (lane & 3))) & 3);
    } else {
      source = ((lane & mask) | ((offset >> 5) & 31)) ^ ((offset >> 10) & 31);
    }
    source = (lane & 32) | (source & 31);
    result[lane] = ((exec_mask >> source) & 1) ? a[0][source] : 0;
  }
  for (unsigned lane = 0; lane < Lanes; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_ds_swizzle_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, uint16_t offset) {
  return swizzle<32>(flags, exec_mask, instruction_flags, d, a, offset);
}

int goc_ds_swizzle_b32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a, uint16_t offset) {
  return swizzle<64>(flags, exec_mask, instruction_flags, d, a, offset);
}

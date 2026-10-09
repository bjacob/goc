// SPDX-License-Identifier: MIT

#ifndef GOC_RDNA4_BITFIELD_REFERENCE_H_
#define GOC_RDNA4_BITFIELD_REFERENCE_H_

#include "goc/goc.h"

#include <stdint.h>

namespace goc_test {

// Per-bit oracle, including signed padding beyond the source's high bit.
inline uint32_t bitfield_reference(int op, uint32_t a, uint32_t b, uint32_t c) {
  uint32_t result = 0;
  unsigned offset = b % 32, width = c % 32;
  for (unsigned bit = 0; bit < 32; ++bit) {
    unsigned value = 0;
    if (op < 2 && width) {
      unsigned source = offset + (bit < width ? bit : width - 1);
      if (bit < width || op == 1)
        value = source < 32 ? (a >> source) & 1 : (op == 1 ? a >> 31 : 0);
    }
    if (op == 2)
      value = ((a >> bit) & 1) ? (b >> bit) & 1 : (c >> bit) & 1;
    if (op == 3)
      value = bit >= offset && bit - offset < a % 32;
    if (op == 4)
      value = (a >> (31 - bit)) & 1;
    if (op == 5 || op == 6) {
      unsigned source = bit + (op == 5 ? c % 32 : (c % 4) * 8);
      value = source < 32 ? (b >> source) & 1 : (a >> (source - 32)) & 1;
    }
    if (op == 7) {
      unsigned selector = (c >> (8 * (bit / 8))) & 255;
      unsigned source = selector < 8    ? 8 * selector + bit % 8
                        : selector < 12 ? 16 * (selector - 8) + 15
                                        : 64;
      value = source < 32   ? (b >> source) & 1
              : source < 64 ? (a >> (source - 32)) & 1
                            : selector > 12;
    }
    result |= uint32_t(value) << bit;
  }
  return result;
}

inline int bitfield_mask(uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *) {
  return goc_rdna4_v_bfm_b32(flags, mask, mode, d, a, b);
}

inline int bitfield_reverse(uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *,
                            const uint32_t *const *) {
  return goc_rdna4_v_bfrev_b32(flags, mask, mode, d, a);
}

} // namespace goc_test

#endif

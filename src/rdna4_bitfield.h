// SPDX-License-Identifier: MIT

#ifndef GOC_RDNA4_BITFIELD_H_
#define GOC_RDNA4_BITFIELD_H_

#include <stdint.h>

namespace goc {

enum class Bitfield {
  ExtractUnsigned,
  ExtractSigned,
  Insert,
  Mask,
  Reverse,
  AlignBit,
  AlignByte,
  Permute
};

template <Bitfield Op>
void bitfield_x86_64_v3(uint32_t mask, uint32_t *d, const uint32_t *a, const uint32_t *b,
                        const uint32_t *c);

template <Bitfield Op>
void bitfield_x86_64_v4(uint32_t mask, uint32_t *d, const uint32_t *a, const uint32_t *b,
                        const uint32_t *c);

} // namespace goc

#endif

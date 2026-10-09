// SPDX-License-Identifier: MIT

#ifndef GOC_RDNA4_SAD_H_
#define GOC_RDNA4_SAD_H_

#include <stdint.h>

namespace goc {

enum class Sad { U8, HighU8, U16, U32, MaskedU8, QuadU16, MaskedQuadU16, MaskedQuadU32 };

constexpr bool sad_quad(Sad op) { return op >= Sad::QuadU16; }

constexpr bool sad_packed(Sad op) { return op == Sad::QuadU16 || op == Sad::MaskedQuadU16; }

constexpr bool sad_masked(Sad op) {
  return op == Sad::MaskedU8 || op == Sad::MaskedQuadU16 || op == Sad::MaskedQuadU32;
}

constexpr int sad_outputs(Sad op) { return op == Sad::MaskedQuadU32 ? 4 : sad_quad(op) ? 2 : 1; }

template <Sad Op>
void sad_x86_64_v3(uint32_t mask, bool clamp, uint32_t *const *d, const uint32_t *const *a,
                   const uint32_t *b, const uint32_t *const *c);

} // namespace goc

#endif

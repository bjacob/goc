// SPDX-License-Identifier: MIT

#pragma once

#include <immintrin.h>
#include <stdint.h>

namespace goc {

// Byte indices select either half independently within every original VGPR word.
inline __m256i packed_integer_selection(bool low_high, bool high_low) {
  uint32_t low = low_high ? 0x0302 : 0x0100;
  uint32_t high = high_low ? 0x0100 : 0x0302;
  return _mm256_add_epi8(_mm256_set1_epi32(int(low | (high << 16))),
                         _mm256_setr_epi32(0, 0x04040404, 0x08080808, 0x0c0c0c0c, 0, 0x04040404,
                                           0x08080808, 0x0c0c0c0c));
}

} // namespace goc

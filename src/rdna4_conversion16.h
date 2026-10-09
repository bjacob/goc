// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc {

enum class Conversion16 {
  SignedToHalf,
  UnsignedToHalf,
  HalfToSigned,
  HalfToUnsigned,
  FloatToHalf,
  HalfToFloat
};

template <Conversion16 Op>
void conversion16_x86_64_v3(bool saturate, uint32_t mask, uint32_t mode, uint32_t *d,
                            const uint32_t *a);

template <Conversion16 Op>
void conversion16_x86_64_v4(bool saturate, uint32_t mask, uint32_t mode, uint32_t *d,
                            const uint32_t *a);

} // namespace goc

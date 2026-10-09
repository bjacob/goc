// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc {

enum class Conversion32 {
  SignedToFloat,
  UnsignedToFloat,
  FloatToSigned,
  FloatToUnsigned,
  Nearest,
  Floor
};

template <Conversion32 Op>
void conversion32_x86_64_v3(uint32_t mask, uint32_t mode, uint32_t *d, const uint32_t *a);

template <Conversion32 Op>
void conversion32_x86_64_v4(uint32_t mask, uint32_t mode, uint32_t *d, const uint32_t *a);

} // namespace goc

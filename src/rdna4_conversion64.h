// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc {

enum class Conversion64 {
  SignedToDouble,
  UnsignedToDouble,
  DoubleToSigned,
  DoubleToUnsigned,
  FloatToDouble,
  DoubleToFloat
};

template <Conversion64 Op>
void conversion64_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                            const uint32_t *const *a);

template <Conversion64 Op>
void conversion64_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                            const uint32_t *const *a);

} // namespace goc

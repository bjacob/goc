// SPDX-License-Identifier: MIT

#pragma once

#include "rdna4_binary.h"

#include <stdint.h>

namespace goc {

template <Binary Op, bool Packed = false>
void half_binary_x86_64_v3(bool saturate, uint32_t mask, uint32_t mode, uint32_t *d,
                           const uint32_t *a, const uint32_t *b);

} // namespace goc

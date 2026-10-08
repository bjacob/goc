// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc {

template <bool Ldexp>
void half_exponent_x86_64_v3(bool saturate, uint32_t mask, uint32_t mode, uint32_t *d,
                             const uint32_t *a, const uint32_t *b);

} // namespace goc

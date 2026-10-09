// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc {

enum class Boolean { And, Or, Xor, Not };

template <Boolean Op, bool Half>
void boolean_x86_64_v4(uint32_t mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                       const uint32_t *b);

} // namespace goc

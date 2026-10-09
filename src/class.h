// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc {

template <unsigned Bits>
uint32_t class_x86_64_v3(uint32_t mode, const uint32_t *const *a, const uint32_t *b);
template <unsigned Bits>
uint32_t class_x86_64_v4(uint32_t mode, const uint32_t *const *a, const uint32_t *b);

} // namespace goc

// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc {

template <bool Bf8, bool Packed>
void fp8_conversion_x86_64_v3(uint32_t mask, unsigned shift, uint32_t *const *d, const uint32_t *a);

template <bool Bf8, bool Packed>
void fp8_conversion_x86_64_v4(uint32_t mask, unsigned shift, uint32_t *const *d, const uint32_t *a);

} // namespace goc

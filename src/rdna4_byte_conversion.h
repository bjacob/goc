// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc {

template <unsigned Byte>
void byte_conversion_x86_64_v3(uint32_t mask, uint32_t mode, uint32_t *d, const uint32_t *a);

template <unsigned Byte>
void byte_conversion_x86_64_v4(uint32_t mask, uint32_t mode, uint32_t *d, const uint32_t *a);

} // namespace goc

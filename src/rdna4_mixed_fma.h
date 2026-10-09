// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc {

enum class MixedFma { Float, Low, High };

void mixed_fma_float_x86_64_v3(uint32_t mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                               const uint32_t *b, const uint32_t *c);

// Write the selected destination half, preserving the other half and inactive lanes.
void mixed_fma_half_x86_64_v3(bool high, bool saturate, uint32_t mask, uint32_t mode, uint32_t *d,
                              const uint32_t *a, const uint32_t *b, const uint32_t *c);

} // namespace goc

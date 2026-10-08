// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc {

void half_fma_x86_64_v3(bool saturate, uint32_t mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                        const uint32_t *b, const uint32_t *c);

} // namespace goc

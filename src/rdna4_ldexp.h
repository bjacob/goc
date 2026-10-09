// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc {

void ldexp_x86_64_v3(bool fp64, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *b);
void ldexp_x86_64_v4(bool fp64, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *b);

} // namespace goc

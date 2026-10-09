// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc {
template <bool Cross, bool Var>
void permlane_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                        const uint32_t *b, uint32_t lo, uint32_t hi);
template <bool Cross, bool Var>
void permlane_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                        const uint32_t *b, uint32_t lo, uint32_t hi);

} // namespace goc

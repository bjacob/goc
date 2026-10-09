// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <stdint.h>

namespace goc {

// Two exponent ranges, each indexed by a clamped table-bit offset.
extern const std::array<uint64_t, 2370> trig_preop_table;

void trig_preop_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                          const uint32_t *a_hi, const uint32_t *b);
void trig_preop_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                          const uint32_t *a_hi, const uint32_t *b);

} // namespace goc

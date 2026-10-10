// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc {

template <bool FirstMaximum, bool SecondMaximum, bool Propagate, bool Median = false,
          bool Rdna3 = false>
void half_minmax3_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                            const uint32_t *a, const uint32_t *b, const uint32_t *c);

} // namespace goc

// SPDX-License-Identifier: MIT

#pragma once

#include "rdna4_fma.h"

#include <stdint.h>

namespace goc {

template <FmaOperands Operands = FmaOperands::Registers>
void half_fma_x86_64_v3(bool saturate, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                        const uint32_t *a, const uint32_t *b, const uint32_t *c,
                        uint16_t literal = 0);

} // namespace goc

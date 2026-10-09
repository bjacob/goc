// SPDX-License-Identifier: MIT

#pragma once

#include "internal.h"

#include <stdint.h>

namespace goc {

enum class BitCount { Leading, Trailing, Sign, Population, MaskedLow, MaskedHigh };

template <BitCount Op, int Lanes>
void bit_count_x86_64_v3(ExecMask<Lanes> exec_mask, uint32_t *d, const uint32_t *a,
                         const uint32_t *b);

template <BitCount Op, int Lanes>
void bit_count_x86_64_v4(ExecMask<Lanes> exec_mask, uint32_t *d, const uint32_t *a,
                         const uint32_t *b);

} // namespace goc

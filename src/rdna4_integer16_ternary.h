// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc {

enum class Integer16Ternary { Mad, Min, Max, Median };

template <bool Signed, Integer16Ternary Op = Integer16Ternary::Mad, bool Packed = true>
void integer16_ternary_x86_64_v3(uint32_t mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                                 const uint32_t *b, const uint32_t *c);

} // namespace goc

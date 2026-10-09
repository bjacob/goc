// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc {

enum class Integer16 { Add, Sub, Min, Max, Mul, ShiftLeft, ShiftRight };

template <Integer16 Op, bool Signed, bool Packed = true>
void integer16_x86_64_v3(uint32_t mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                         const uint32_t *b);

} // namespace goc

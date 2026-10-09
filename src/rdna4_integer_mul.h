// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc {

template <int Bits, bool Signed, bool High>
void integer_mul_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                           const uint32_t *b);

template <int Bits, bool Signed, bool High>
void integer_mul_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                           const uint32_t *b);

} // namespace goc

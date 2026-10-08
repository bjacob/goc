// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc {

enum class IntegerAdd { Add, Sub, Subrev, Add3 };

template <IntegerAdd Op, bool Signed>
void integer_add_sat_x86_64_v3(uint32_t mask, uint32_t *d, const uint32_t *a, const uint32_t *b);

template <IntegerAdd Op, bool Signed>
void integer_add_x86_64_v4(uint32_t mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                           const uint32_t *b, const uint32_t *c);

} // namespace goc

// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc {

enum class IntegerMinmax { Min, Max, Min3, Max3, Minmax, Maxmin, Median };

template <IntegerMinmax Op, bool Signed>
void integer_minmax_x86_64_v3(uint32_t mask, uint32_t *d, const uint32_t *a, const uint32_t *b,
                              const uint32_t *c);

template <IntegerMinmax Op, bool Signed>
void integer_minmax_x86_64_v4(uint32_t mask, uint32_t *d, const uint32_t *a, const uint32_t *b,
                              const uint32_t *c);

} // namespace goc

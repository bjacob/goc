// SPDX-License-Identifier: MIT

#pragma once

#include "unary.h"

#include <stdint.h>

namespace goc {

// RCP/RSQ/SQRT use the captured FP32 approximation followed by half narrowing.
template <Unary Op>
constexpr bool half_unary_sfu = Op == Unary::Rcp || Op == Unary::Rsq || Op == Unary::Sqrt;

template <Unary Op>
void half_unary_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                          const uint32_t *a);

} // namespace goc

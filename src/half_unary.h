// SPDX-License-Identifier: MIT

#pragma once

#include "unary.h"

#include <stdint.h>

namespace goc {

template <Unary Op>
void half_unary_x86_64_v3(bool saturate, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                          const uint32_t *a);

} // namespace goc

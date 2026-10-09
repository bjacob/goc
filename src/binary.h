// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc {

enum class Binary { Add, Sub, Subrev, Mul, MulDx9Zero, MinNum, MaxNum, Minimum, Maximum };

#if defined(GOC_HAVE_X86_64_V3)
void binary_x86_64_v3(Binary op, uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                      const uint32_t *b);
#endif

} // namespace goc

// SPDX-License-Identifier: MIT

#ifndef GOC_RDNA4_SHIFT_H_
#define GOC_RDNA4_SHIFT_H_

#include <stdint.h>

namespace goc {

enum class Shift { Left, LogicalRight, ArithmeticRight };

template <int Bits, Shift Op>
void shift_x86_64_v3(uint32_t exec_mask, uint32_t *const *d, const uint32_t *a,
                     const uint32_t *const *b);

} // namespace goc

#endif

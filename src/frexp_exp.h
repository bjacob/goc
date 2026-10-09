// SPDX-License-Identifier: MIT

#ifndef GOC_FREXP_EXP_H_
#define GOC_FREXP_EXP_H_

#include <stdint.h>

namespace goc {

void frexp_exp_x86_64_v3(bool fp64, uint32_t exec_mask, uint32_t *d, const uint32_t *const *a);

} // namespace goc

#endif

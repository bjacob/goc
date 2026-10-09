// SPDX-License-Identifier: MIT

#ifndef GOC_TRIG_H_
#define GOC_TRIG_H_

#include <stdint.h>

namespace goc {

void trig_x86_64_v3(bool cosine, uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a);

void half_trig_x86_64_v3(bool cosine, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                         const uint32_t *a);

} // namespace goc

#endif

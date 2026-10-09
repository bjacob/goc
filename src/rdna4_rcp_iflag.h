// SPDX-License-Identifier: MIT

#ifndef GOC_RDNA4_RCP_IFLAG_H_
#define GOC_RDNA4_RCP_IFLAG_H_

#include <stdint.h>

namespace goc {

uint32_t rcp_iflag_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a);
uint32_t rcp_iflag_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a);

} // namespace goc

#endif

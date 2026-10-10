// SPDX-License-Identifier: MIT

#include "dpp.h"
#include "goc/goc.h"
#include "internal.h"

#include <stdint.h>

int goc_v_mov_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a) {
  if (instruction_flags >> 32)
    return goc::execute_dpp(flags, exec_mask, instruction_flags, a,
                            [&](uint32_t exec_mask, const uint32_t *const *source) {
                              return goc_v_mov_b32(flags, exec_mask, uint32_t(instruction_flags), d,
                                                   source);
                            });
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
  for (unsigned lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = a[0][lane];
  return GOC_SUCCESS;
}

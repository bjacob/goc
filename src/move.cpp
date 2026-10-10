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

int goc_v_mov_b16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a) {
  if (instruction_flags >> 32)
    return goc::execute_dpp(flags, exec_mask, instruction_flags, a,
                            [&](uint32_t exec_mask, const uint32_t *const *source) {
                              return goc_v_mov_b16(flags, exec_mask, uint32_t(instruction_flags), d,
                                                   source);
                            });
  const uint32_t known = GOC_ALU_HIGH_A | GOC_ALU_HIGH_D | GOC_ALU_NEG_A | GOC_ALU_ABS_A |
                         GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP;
  if (int error = goc::validate(flags, instruction_flags & ~known, true))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
  unsigned sa = instruction_flags & GOC_ALU_HIGH_A ? 16 : 0;
  unsigned sd = instruction_flags & GOC_ALU_HIGH_D ? 16 : 0;
  for (unsigned lane = 0; lane < 32; ++lane) {
    uint32_t value = (a[0][lane] >> sa) & 65535;
    if (instruction_flags & GOC_ALU_ABS_A)
      value &= 32767;
    if (instruction_flags & GOC_ALU_NEG_A)
      value ^= 32768;
    // MOV modifies the sign bit but does not perform FP arithmetic. Hardware
    // ignores OMOD and CLAMP, including for subnormals and signaling NaNs.
    if ((exec_mask >> lane) & 1)
      d[0][lane] = (d[0][lane] & ~(65535U << sd)) | (value << sd);
  }
  return GOC_SUCCESS;
}

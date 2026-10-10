// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "internal.h"

#include <stdint.h>

namespace {

template <unsigned Shift>
int set_mode(uint64_t flags, uint64_t instruction_flags, uint16_t immediate, uint32_t *mode) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, uint32_t(instruction_flags), true))
    return error;
  if (mode)
    *mode = (*mode & ~(15U << Shift)) | (uint32_t(immediate & 15) << Shift);
  return GOC_SUCCESS;
}

} // namespace

int goc_s_round_mode(uint64_t flags, uint64_t instruction_flags, uint16_t immediate,
                     uint32_t *mode) {
  return set_mode<0>(flags, instruction_flags, immediate, mode);
}

int goc_s_denorm_mode(uint64_t flags, uint64_t instruction_flags, uint16_t immediate,
                      uint32_t *mode) {
  return set_mode<4>(flags, instruction_flags, immediate, mode);
}

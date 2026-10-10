// SPDX-License-Identifier: MIT

// Register transfer semantics follow rocjitsu generated/shared/execute_shared.h.

#include "goc/goc.h"
#include "internal.h"

#include <stdint.h>

namespace {

template <typename T> int move(uint64_t flags, uint64_t mode, T *d, T a, bool write) {
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, uint32_t(mode), true))
    return error;
  if (write)
    *d = a;
  return GOC_SUCCESS;
}

uint32_t signed_immediate(uint16_t immediate) {
  return uint32_t(immediate) | ((immediate & 0x8000U) ? 0xffff0000U : 0);
}

} // namespace

int goc_s_mov_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a) {
  return move(flags, instruction_flags, d, a, true);
}

int goc_s_mov_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a) {
  return move(flags, instruction_flags, d, a, true);
}

int goc_s_cmov_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                   uint32_t input_scc) {
  return move(flags, instruction_flags, d, a, (input_scc & 1) != 0);
}

int goc_s_cmov_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                   uint32_t input_scc) {
  return move(flags, instruction_flags, d, a, (input_scc & 1) != 0);
}

int goc_s_movk_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint16_t immediate) {
  return move(flags, instruction_flags, d, signed_immediate(immediate), true);
}

int goc_s_cmovk_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint16_t immediate,
                    uint32_t input_scc) {
  return move(flags, instruction_flags, d, signed_immediate(immediate), (input_scc & 1) != 0);
}

int goc_s_getpc_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t pc) {
  return goc_s_mov_b64(flags, instruction_flags, d, pc + 4);
}

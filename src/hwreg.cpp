// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "internal.h"

#include <stdint.h>

int goc_s_setreg_b32_rdna4(uint64_t flags, uint64_t instruction_flags, uint16_t hwreg, uint32_t a,
                           uint32_t status, uint32_t *mode, uint32_t *state_priv,
                           uint32_t *excp_flag_priv, uint32_t *excp_flag_user, uint32_t *trap_ctrl,
                           uint64_t *scratch_base) {
  unsigned id = hwreg & 63;
  bool privileged = status & (1U << 5);
  // The alignment requirement does not establish the scratch write mask or
  // whether writing it affects STATE_PRIV.SCRATCH_EN. Do not invent updates.
  if ((id == 20 || id == 21) && privileged && (scratch_base || state_priv) &&
      (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, uint32_t(instruction_flags), true))
    return error;

  unsigned offset = (hwreg >> 6) & 31;
  unsigned width = (hwreg >> 11) + 1;
  // A 64-bit intermediate handles width 32 and clips fields crossing bit 31.
  uint32_t field_mask = uint32_t(((1ULL << width) - 1) << offset);
  uint32_t *destination = nullptr;
  uint32_t writable = 0;
  switch (id) {
  case 1:
    destination = mode;
    // FP modes [7:0], FP16_OVFL [23], scalar prefetch [24], DISABLE_PERF [27].
    writable = 0x098000ff;
    break;
  case 4:
    destination = state_priv;
    // WG/sleep/barrier [2:0], SCC/priorities/HALT [14:9], SCRATCH_EN [18].
    writable = privileged ? 0x00047e07 : 0;
    break;
  case 17:
    destination = excp_flag_priv;
    // Exception/status flags [11:0], first memory-violation source [31:30].
    writable = privileged ? 0xc0000fff : 0;
    break;
  case 18:
    destination = excp_flag_user;
    // Arithmetic [6:0], buffer OOB [30], LOD clamped [31].
    writable = 0xc000007f;
    break;
  case 19:
    destination = trap_ctrl;
    // Arithmetic, address-watch, wave-end and trap-after-instruction enables.
    writable = privileged ? 0x000003ff : 0;
    break;
  case 20:
  case 21:
    // Null opts out; loose mode skips unavailable scratch-state reporting.
    return GOC_SUCCESS;
  // STATUS, allocation state, snapshots, hardware IDs, IB_STS2 and clock halves.
  case 2:
  case 5:
  case 6:
  case 10:
  case 11:
  case 12:
  case 15:
  case 16:
  case 23:
  case 24:
  case 28:
  case 29:
  case 30:
    return GOC_SUCCESS;
  default:
    return GOC_ERROR_INVALID_FLAGS;
  }
  uint32_t mask = field_mask & writable;
  if (mask && destination)
    *destination = (*destination & ~mask) | ((a << offset) & mask);
  return GOC_SUCCESS;
}

int goc_s_setreg_imm32_b32_rdna4(uint64_t flags, uint64_t instruction_flags, uint16_t hwreg,
                                 uint32_t immediate, uint32_t status, uint32_t *mode,
                                 uint32_t *state_priv, uint32_t *excp_flag_priv,
                                 uint32_t *excp_flag_user, uint32_t *trap_ctrl,
                                 uint64_t *scratch_base) {
  return goc_s_setreg_b32_rdna4(flags, instruction_flags, hwreg, immediate, status, mode,
                                state_priv, excp_flag_priv, excp_flag_user, trap_ctrl,
                                scratch_base);
}

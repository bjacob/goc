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
  const uint32_t known = GOC_ALU_ABS_A | GOC_ALU_NEG_A;
  if (int error = goc::validate(flags, instruction_flags & ~known, true))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
  if (instruction_flags) {
    // ABS precedes NEG; all other bits, including NaN payloads, are preserved.
    uint32_t keep = instruction_flags & GOC_ALU_ABS_A ? 0x7fffffffU : UINT32_MAX;
    uint32_t toggle = instruction_flags & GOC_ALU_NEG_A ? 0x80000000U : 0;
    for (unsigned lane = 0; lane < 32; ++lane)
      if ((exec_mask >> lane) & 1)
        d[0][lane] = (a[0][lane] & keep) ^ toggle;
    return GOC_SUCCESS;
  }
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

namespace {

// Accepted DPP controls never cross a 16-lane row, so each 32-lane half can
// reuse the existing implementation. Whole-VGPR aliasing cannot couple halves.
template <bool Half>
int move_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                const uint32_t *const *a) {
  constexpr auto move = Half ? goc_v_mov_b16 : goc_v_mov_b32;
  if (int error = move(flags, 0, mode, nullptr, nullptr))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
  for (unsigned half = 0; half < 2; ++half) {
    uint64_t half_mode = mode;
    if (mode & GOC_DPP16) {
      uint64_t rows = (mode & GOC_DPP_ROW_MASK) >> GOC_DPP_ROW_SHIFT;
      half_mode = (mode & ~GOC_DPP_ROW_MASK) | (((rows >> (half * 2)) & 3) << GOC_DPP_ROW_SHIFT);
    }
    uint32_t *destination = d[0] + half * 32;
    const uint32_t *source = a[0] + half * 32;
    // Validation above covers both calls: only valid row-enable bits differ.
    int error = move(flags, uint32_t(exec_mask >> (half * 32)), half_mode, &destination, &source);
    if (error)
      return error;
  }
  return GOC_SUCCESS;
}

} // namespace

int goc_v_mov_b32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a) {
  return move_wave64<false>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_mov_b16_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a) {
  return move_wave64<true>(flags, exec_mask, instruction_flags, d, a);
}

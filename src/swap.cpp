// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "internal.h"

#include <stdint.h>

namespace {

template <unsigned Lanes>
int swap(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
         uint32_t *const *a) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, uint32_t(instruction_flags), true))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
  for (unsigned lane = 0; lane < Lanes; ++lane) {
    uint32_t old_d = d[0][lane], old_a = a[0][lane];
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = old_a;
      a[0][lane] = old_d;
    }
  }
  return GOC_SUCCESS;
}

} // namespace

int goc_v_swap_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                   uint32_t *const *d, uint32_t *const *a) {
  return swap<32>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_swap_b32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                          uint32_t *const *d, uint32_t *const *a) {
  return swap<64>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_permlane64_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *const *,
                         const uint32_t *const *) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return goc::validate(flags, uint32_t(instruction_flags), true);
}

int goc_v_permlane64_b32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, uint32_t(instruction_flags), true))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
  // Snapshot both halves before stores so D can alias A.
  uint32_t result[64];
  for (unsigned lane = 0; lane < 64; ++lane)
    result[lane] = a[0][lane ^ 32];
  for (unsigned lane = 0; lane < 64; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

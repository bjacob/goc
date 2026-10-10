// SPDX-License-Identifier: MIT

// Scalar lane-transfer rules follow rocjitsu generated RDNA3/RDNA4 execution
// and vm/amdgpu/register_access.h (lane indices wrap to the wave size).

#include "bits.h"
#include "goc/goc.h"
#include "internal.h"

#include <stdint.h>

namespace {

int validate(uint64_t flags, uint64_t mode) {
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return goc::validate(flags, uint32_t(mode), true);
}

} // namespace

int goc_v_readfirstlane_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a) {
  if (int error = validate(flags, instruction_flags))
    return error;
  unsigned lane = exec_mask ? goc::bit_count_zero<true>(exec_mask) : 0;
  *d = a[0][lane];
  return GOC_SUCCESS;
}

int goc_v_readlane_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                       const uint32_t *const *a, uint32_t lane) {
  if (int error = validate(flags, instruction_flags))
    return error;
  *d = a[0][lane & 31];
  return GOC_SUCCESS;
}

int goc_v_writelane_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d, uint32_t a,
                        uint32_t lane) {
  if (int error = validate(flags, instruction_flags))
    return error;
  d[0][lane & 31] = a;
  return GOC_SUCCESS;
}

int goc_v_readfirstlane_b32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *d, const uint32_t *const *a) {
  if (int error = validate(flags, instruction_flags))
    return error;
  unsigned lane = exec_mask ? goc::bit_count_zero<true>(exec_mask) : 0;
  *d = a[0][lane];
  return GOC_SUCCESS;
}

int goc_v_readlane_b32_wave64(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                              const uint32_t *const *a, uint32_t lane) {
  if (int error = validate(flags, instruction_flags))
    return error;
  *d = a[0][lane & 63];
  return GOC_SUCCESS;
}

int goc_v_writelane_b32_wave64(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                               uint32_t a, uint32_t lane) {
  if (int error = validate(flags, instruction_flags))
    return error;
  d[0][lane & 63] = a;
  return GOC_SUCCESS;
}

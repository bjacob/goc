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

// Both operands are destinations: an out-of-range index discards the entire
// instruction. Keep widened sums so a large base cannot wrap into the file.
template <unsigned Lanes>
int swap_relative(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                  uint32_t *const *a, uint32_t d_count, uint32_t a_count, uint32_t d_base,
                  uint32_t a_base, uint32_t m0) {
  // Reject flags before forming pointers into either register-file view.
  if (int error = swap<Lanes>(flags, 0, mode, nullptr, nullptr))
    return error;
  uint32_t src_offset = m0 & 1023;
  uint32_t dst_offset = (m0 >> 16) & 1023;
  uint64_t src = uint64_t(a_base) + src_offset;
  uint64_t dst = uint64_t(d_base) + dst_offset;
  if (!exec_mask || src_offset > 255 || dst_offset > 255 || src >= a_count || dst >= d_count)
    return swap<Lanes>(flags, 0, mode, nullptr, nullptr);
  return swap<Lanes>(flags, exec_mask, mode, d + dst, a + src);
}

template <unsigned Lanes>
int swap_half(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
              uint32_t *const *a) {
  const uint64_t known = GOC_ALU_HIGH_A | GOC_ALU_HIGH_D;
  if (mode & ~known)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, 0, true))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
  unsigned sa = mode & GOC_ALU_HIGH_A ? 16 : 0;
  unsigned sd = mode & GOC_ALU_HIGH_D ? 16 : 0;
  for (unsigned lane = 0; lane < Lanes; ++lane) {
    uint32_t old_d = (d[0][lane] >> sd) & 65535;
    uint32_t old_a = (a[0][lane] >> sa) & 65535;
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = (d[0][lane] & ~(65535U << sd)) | (old_a << sd);
      // Preserve the first write if A and D are halves of the same VGPR.
      a[0][lane] = (a[0][lane] & ~(65535U << sa)) | (old_d << sa);
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

int goc_v_swap_b16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                   uint32_t *const *d, uint32_t *const *a) {
  return swap_half<32>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_swap_b16_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                          uint32_t *const *d, uint32_t *const *a) {
  return swap_half<64>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_swaprel_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, uint32_t *const *a, uint32_t d_count, uint32_t a_count,
                      uint32_t d_base, uint32_t a_base, uint32_t m0) {
  return swap_relative<32>(flags, exec_mask, instruction_flags, d, a, d_count, a_count, d_base,
                           a_base, m0);
}

int goc_v_swaprel_b32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, uint32_t *const *a, uint32_t d_count,
                             uint32_t a_count, uint32_t d_base, uint32_t a_base, uint32_t m0) {
  return swap_relative<64>(flags, exec_mask, instruction_flags, d, a, d_count, a_count, d_base,
                           a_base, m0);
}

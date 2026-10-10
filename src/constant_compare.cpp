// SPDX-License-Identifier: MIT

// RDNA3 constant integer predicates follow rocjitsu execute_v_cmp_[ft]_*.

#include "dpp.h"
#include "goc/goc.h"
#include "internal.h"

#include <stdint.h>

namespace {

int compare(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d, bool value, bool wide) {
  if (mode >> 32) {
    if (wide || !(mode & (GOC_DPP8 | GOC_DPP16)) || !goc::valid_dpp(mode))
      return GOC_ERROR_INVALID_FLAGS;
  }
  if (int error = goc::validate(flags, uint32_t(mode), true))
    return error;
  if (value && exec_mask && (mode & GOC_DPP16)) {
    // Operand values are irrelevant, but DPP row/bank/boundary filtering is not.
    uint32_t zero[32] = {}, ignored[32];
    exec_mask = goc::dpp16_source(flags, exec_mask, mode, ignored, zero);
  }
  *d = value ? exec_mask : 0;
  return GOC_SUCCESS;
}

} // namespace

int goc_v_cmp_f_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, false, false);
}

int goc_v_cmp_t_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, true, false);
}

int goc_v_cmpx_f_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, false, false);
}

int goc_v_cmpx_t_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, true, false);
}

int goc_v_cmp_f_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, false, false);
}

int goc_v_cmp_t_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, true, false);
}

int goc_v_cmpx_f_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, false, false);
}

int goc_v_cmpx_t_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, true, false);
}

int goc_v_cmp_f_i64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, false, true);
}

int goc_v_cmp_t_i64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, true, true);
}

int goc_v_cmpx_f_i64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, false, true);
}

int goc_v_cmpx_t_i64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, true, true);
}

int goc_v_cmp_f_u64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, false, true);
}

int goc_v_cmp_t_u64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, true, true);
}

int goc_v_cmpx_f_u64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, false, true);
}

int goc_v_cmpx_t_u64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, true, true);
}

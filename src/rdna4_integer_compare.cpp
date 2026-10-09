// SPDX-License-Identifier: MIT

// Comparisons follow rocjitsu's execute_v_cmp_* integer models.

#include "rdna4_integer_compare.h"
#include "goc/goc.h"
#include "internal.h"

#include <stdint.h>

namespace {

template <unsigned Bits, bool Signed, unsigned Predicate>
int run(uint64_t flags, uint64_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *const *a,
        const uint32_t *const *b) {
  const uint32_t known = Bits == 16 ? GOC_ALU_HIGH_A | GOC_ALU_HIGH_B : 0;
  if (int error = goc::validate(flags, mode & ~known, true))
    return error;
  const uint32_t mask = uint32_t(exec_mask);
  if (!mask) {
    *d = 0;
    return GOC_SUCCESS;
  }
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    *d = goc::integer_compare_x86_64_v4<Bits, Signed, Predicate>(mode, a, b) & mask;
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    *d = goc::integer_compare_x86_64_v3<Bits, Signed, Predicate>(mode, a, b) & mask;
    return GOC_SUCCESS;
  }
#endif
  uint32_t less = 0, equal = 0;
  // Baseline x86 vectorized variable shifts can raise host FP exceptions.
#if defined(__clang__)
#pragma clang loop vectorize(disable)
#endif
  for (unsigned lane = 0; lane < 32; ++lane) {
    uint64_t av = a[0][lane], bv = b[0][lane];
    if constexpr (Bits == 16) {
      av = (av >> (mode & GOC_ALU_HIGH_A ? 16 : 0)) & 65535;
      bv = (bv >> (mode & GOC_ALU_HIGH_B ? 16 : 0)) & 65535;
    }
    if constexpr (Bits == 64) {
      av |= uint64_t(a[1][lane]) << 32;
      bv |= uint64_t(b[1][lane]) << 32;
    }
    if constexpr (Signed) {
      av ^= UINT64_C(1) << (Bits - 1);
      bv ^= UINT64_C(1) << (Bits - 1);
    }
    less |= uint32_t(av < bv) << lane;
    equal |= uint32_t(av == bv) << lane;
  }
  *d = goc::integer_compare_result<Predicate>(less, equal) & mask;
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_cmp_lt_i16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, true, 1>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_eq_i16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, true, 2>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_le_i16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, true, 3>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_gt_i16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, true, 4>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_ne_i16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, true, 5>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_ge_i16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, true, 6>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_lt_i16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, true, 1>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_eq_i16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, true, 2>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_le_i16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, true, 3>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_gt_i16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, true, 4>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_ne_i16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, true, 5>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_ge_i16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, true, 6>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_lt_u16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, false, 1>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_eq_u16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, false, 2>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_le_u16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, false, 3>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_gt_u16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, false, 4>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_ne_u16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, false, 5>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_ge_u16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, false, 6>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_lt_u16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, false, 1>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_eq_u16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, false, 2>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_le_u16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, false, 3>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_gt_u16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, false, 4>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_ne_u16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, false, 5>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_ge_u16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, false, 6>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_lt_i32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, true, 1>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_eq_i32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, true, 2>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_le_i32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, true, 3>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_gt_i32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, true, 4>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_ne_i32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, true, 5>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_ge_i32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, true, 6>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_lt_i32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, true, 1>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_eq_i32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, true, 2>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_le_i32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, true, 3>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_gt_i32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, true, 4>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_ne_i32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, true, 5>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_ge_i32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, true, 6>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_lt_u32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, false, 1>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_eq_u32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, false, 2>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_le_u32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, false, 3>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_gt_u32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, false, 4>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_ne_u32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, false, 5>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_ge_u32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, false, 6>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_lt_u32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, false, 1>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_eq_u32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, false, 2>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_le_u32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, false, 3>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_gt_u32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, false, 4>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_ne_u32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, false, 5>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_ge_u32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, false, 6>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_lt_i64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, true, 1>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_eq_i64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, true, 2>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_le_i64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, true, 3>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_gt_i64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, true, 4>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_ne_i64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, true, 5>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_ge_i64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, true, 6>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_lt_i64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, true, 1>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_eq_i64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, true, 2>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_le_i64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, true, 3>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_gt_i64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, true, 4>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_ne_i64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, true, 5>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_ge_i64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, true, 6>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_lt_u64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, false, 1>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_eq_u64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, false, 2>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_le_u64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, false, 3>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_gt_u64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, false, 4>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_ne_u64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, false, 5>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_ge_u64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, false, 6>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_lt_u64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, false, 1>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_eq_u64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, false, 2>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_le_u64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, false, 3>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_gt_u64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, false, 4>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_ne_u64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, false, 5>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_ge_u64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, false, 6>(flags, exec_mask, instruction_flags, d, a, b);
}

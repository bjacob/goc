// SPDX-License-Identifier: MIT

// Raw ordering, signed-zero normalization and predicate inversion follow
// rocjitsu's shared/comparison.h, with guest input flushing before ordering.

#include "rdna4_float_compare.h"
#include "goc/goc.h"
#include "internal.h"

#include <stdint.h>

namespace {

template <unsigned Bits, unsigned Predicate>
int run(uint64_t flags, uint64_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *const *a,
        const uint32_t *const *b) {
  const uint32_t known = GOC_ALU_ABS_A | GOC_ALU_ABS_B | GOC_ALU_NEG_A | GOC_ALU_NEG_B |
                         (Bits == 16 ? GOC_ALU_HIGH_A | GOC_ALU_HIGH_B : 0);
  if (int error = goc::validate(flags, mode & ~known, true,
                                GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS))
    return error;
  const uint32_t mask = uint32_t(exec_mask);
  if (!mask) {
    *d = 0;
    return GOC_SUCCESS;
  }
  bool flush = flags & GOC_FP_FLUSH_INPUT_DENORMALS;
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    *d = goc::float_compare_x86_64_v4<Bits, Predicate>(mode, flush, a, b) & mask;
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    *d = goc::float_compare_x86_64_v3<Bits, Predicate>(mode, flush, a, b) & mask;
    return GOC_SUCCESS;
  }
#endif
  constexpr uint64_t sign = UINT64_C(1) << (Bits - 1), magnitude = sign - 1;
  constexpr uint64_t infinity = Bits == 16   ? 0x7c00
                                : Bits == 32 ? 0x7f800000
                                             : UINT64_C(0x7ff0000000000000);
  uint32_t less = 0, equal = 0, ordered = 0;
  // Keep baseline variable shifts from introducing host FP exceptions.
#if defined(__clang__)
#pragma clang loop vectorize(disable)
#endif
  for (unsigned lane = 0; lane < 32; ++lane) {
    uint64_t av = a[0][lane], bv = b[0][lane];
    if constexpr (Bits == 64) {
      av |= uint64_t(a[1][lane]) << 32;
      bv |= uint64_t(b[1][lane]) << 32;
    }
    if constexpr (Bits == 16) {
      av = (av >> (mode & GOC_ALU_HIGH_A ? 16 : 0)) & 65535;
      bv = (bv >> (mode & GOC_ALU_HIGH_B ? 16 : 0)) & 65535;
    }
    if (mode & GOC_ALU_ABS_A)
      av &= magnitude;
    if (mode & GOC_ALU_ABS_B)
      bv &= magnitude;
    if (mode & GOC_ALU_NEG_A)
      av ^= sign;
    if (mode & GOC_ALU_NEG_B)
      bv ^= sign;
    if (flush) {
      if (!(av & infinity))
        av &= sign;
      if (!(bv & infinity))
        bv &= sign;
    }
    bool ord = (av & magnitude) <= infinity && (bv & magnitude) <= infinity;
    if (!(av & magnitude))
      av = 0;
    if (!(bv & magnitude))
      bv = 0;
    av ^= av & sign ? (sign | magnitude) : sign;
    bv ^= bv & sign ? (sign | magnitude) : sign;
    less |= uint32_t(av < bv) << lane;
    equal |= uint32_t(av == bv) << lane;
    ordered |= uint32_t(ord) << lane;
  }
  *d = goc::float_compare_result<Predicate>(less, equal, ordered) & mask;
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_cmp_lt_f16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, 1>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_eq_f16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, 2>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_le_f16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, 3>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_gt_f16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, 4>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_lg_f16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, 5>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_ge_f16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, 6>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_o_f16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                          uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, 7>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_u_f16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                          uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, 8>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_nge_f16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, 9>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_nlg_f16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, 10>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_ngt_f16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, 11>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_nle_f16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, 12>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_neq_f16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, 13>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_nlt_f16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, 14>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_lt_f16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, 1>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_eq_f16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, 2>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_le_f16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, 3>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_gt_f16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, 4>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_lg_f16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, 5>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_ge_f16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, 6>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_o_f16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, 7>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_u_f16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, 8>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_nge_f16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, 9>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_nlg_f16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, 10>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_ngt_f16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, 11>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_nle_f16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, 12>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_neq_f16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, 13>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_nlt_f16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<16, 14>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_lt_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, 1>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_eq_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, 2>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_le_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, 3>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_gt_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, 4>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_lg_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, 5>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_ge_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, 6>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_o_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                          uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, 7>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_u_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                          uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, 8>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_nge_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, 9>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_nlg_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, 10>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_ngt_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, 11>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_nle_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, 12>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_neq_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, 13>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_nlt_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, 14>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_lt_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, 1>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_eq_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, 2>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_le_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, 3>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_gt_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, 4>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_lg_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, 5>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_ge_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, 6>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_o_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, 7>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_u_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, 8>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_nge_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, 9>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_nlg_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, 10>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_ngt_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, 11>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_nle_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, 12>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_neq_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, 13>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_nlt_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, 14>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_lt_f64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, 1>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_eq_f64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, 2>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_le_f64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, 3>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_gt_f64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, 4>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_lg_f64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, 5>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_ge_f64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, 6>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_o_f64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                          uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, 7>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_u_f64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                          uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, 8>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_nge_f64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, 9>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_nlg_f64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, 10>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_ngt_f64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, 11>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_nle_f64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, 12>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_neq_f64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, 13>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_nlt_f64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, 14>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_lt_f64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, 1>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_eq_f64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, 2>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_le_f64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, 3>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_gt_f64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, 4>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_lg_f64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, 5>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_ge_f64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, 6>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_o_f64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, 7>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_u_f64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, 8>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_nge_f64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, 9>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_nlg_f64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, 10>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_ngt_f64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, 11>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_nle_f64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, 12>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_neq_f64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, 13>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_nlt_f64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, 14>(flags, exec_mask, instruction_flags, d, a, b);
}

// SPDX-License-Identifier: MIT

// Raw ordering, signed-zero normalization and predicate inversion follow
// rocjitsu's shared/comparison.h, with guest input flushing before ordering.

#include "float_compare.h"
#include "dpp.h"
#include "goc/goc.h"
#include "internal.h"

#include <stdint.h>

namespace {

template <unsigned Bits, unsigned Predicate>
int run(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d, const uint32_t *const *a,
        const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (mode >> 32) {
    if constexpr (Bits == 64)
      return GOC_ERROR_INVALID_FLAGS;
    if (!(mode & (GOC_DPP8 | GOC_DPP16)) || !goc::valid_dpp(mode))
      return GOC_ERROR_INVALID_FLAGS;
  }
  const uint32_t known = GOC_ALU_CLAMP | GOC_ALU_ABS_A | GOC_ALU_ABS_B | GOC_ALU_NEG_A |
                         GOC_ALU_NEG_B | (Bits == 16 ? GOC_ALU_HIGH_A | GOC_ALU_HIGH_B : 0);
  if (int error = goc::validate(flags, uint32_t(mode) & ~known, true,
                                GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS))
    return error;
  if (!exec_mask) {
    *d = 0;
    return GOC_SUCCESS;
  }
  if constexpr (Bits != 64) {
    if (mode >> 32) {
      uint32_t permuted[32];
      const uint32_t *source = permuted;
      if (mode & GOC_DPP8)
        goc::dpp8_source(flags, exec_mask, mode, permuted, a[0]);
      else
        exec_mask = goc::dpp16_source(flags, exec_mask, mode, permuted, a[0]);
      return run<Bits, Predicate>(flags, exec_mask, uint32_t(mode), d, &source, b, excp_flag_user);
    }
  }
  bool flush = flags & GOC_FP_FLUSH_INPUT_DENORMALS;
  const bool report =
      excp_flag_user && (flags & GOC_SEMANTICS_MASK) == GOC_SEMANTICS_EXACT_EMPIRICAL;
#if defined(GOC_HAVE_X86_64_V4)
  if (!report && (flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    *d = goc::float_compare_x86_64_v4<Bits, Predicate>(mode, flush, a, b) & exec_mask;
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if (!report && (flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    *d = goc::float_compare_x86_64_v3<Bits, Predicate>(mode, flush, a, b) & exec_mask;
    return GOC_SUCCESS;
  }
#endif
  constexpr uint64_t sign = 1ULL << (Bits - 1), magnitude = sign - 1;
  constexpr uint64_t infinity = Bits == 16   ? 0x7c00
                                : Bits == 32 ? 0x7f800000
                                             : 0x7ff0000000000000ULL;
  uint32_t less = 0, equal = 0, ordered = 0, exceptions = 0;
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
    if (report)
      exceptions |= goc::float_compare_exceptions<Bits>(av, bv, flush, mode & GOC_ALU_CLAMP) *
                    ((exec_mask >> lane) & 1);
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
  *d = goc::float_compare_result<Predicate>(less, equal, ordered) & exec_mask;
  if (report)
    *excp_flag_user |= exceptions;
  return GOC_SUCCESS;
}

} // namespace

int goc_v_cmp_lt_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<16, 1>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_eq_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<16, 2>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_le_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<16, 3>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_gt_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<16, 4>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_lg_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<16, 5>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_ge_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<16, 6>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_o_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<16, 7>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_u_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<16, 8>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_nge_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<16, 9>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_nlg_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<16, 10>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_ngt_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<16, 11>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_nle_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<16, 12>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_neq_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<16, 13>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_nlt_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<16, 14>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_lt_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<16, 1>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_eq_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<16, 2>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_le_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<16, 3>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_gt_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<16, 4>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_lg_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<16, 5>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_ge_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<16, 6>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_o_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<16, 7>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_u_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<16, 8>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_nge_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<16, 9>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_nlg_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<16, 10>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_ngt_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<16, 11>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_nle_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<16, 12>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_neq_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<16, 13>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_nlt_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<16, 14>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_lt_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<32, 1>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_eq_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<32, 2>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_le_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<32, 3>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_gt_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<32, 4>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_lg_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<32, 5>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_ge_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<32, 6>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_o_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<32, 7>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_u_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<32, 8>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_nge_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<32, 9>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_nlg_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<32, 10>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_ngt_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<32, 11>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_nle_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<32, 12>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_neq_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<32, 13>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_nlt_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<32, 14>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_lt_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<32, 1>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_eq_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<32, 2>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_le_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<32, 3>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_gt_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<32, 4>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_lg_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<32, 5>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_ge_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<32, 6>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_o_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<32, 7>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_u_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<32, 8>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_nge_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<32, 9>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_nlg_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<32, 10>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_ngt_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<32, 11>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_nle_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<32, 12>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_neq_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<32, 13>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_nlt_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<32, 14>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_lt_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<64, 1>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_eq_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<64, 2>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_le_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<64, 3>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_gt_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<64, 4>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_lg_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<64, 5>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_ge_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<64, 6>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_o_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<64, 7>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_u_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<64, 8>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_nge_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<64, 9>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_nlg_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<64, 10>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_ngt_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<64, 11>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_nle_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<64, 12>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_neq_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<64, 13>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmp_nlt_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<64, 14>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_lt_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<64, 1>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_eq_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<64, 2>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_le_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<64, 3>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_gt_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<64, 4>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_lg_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<64, 5>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_ge_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<64, 6>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_o_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<64, 7>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_u_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<64, 8>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_nge_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<64, 9>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_nlg_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<64, 10>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_ngt_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<64, 11>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_nle_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<64, 12>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_neq_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<64, 13>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

int goc_v_cmpx_nlt_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<64, 14>(flags, exec_mask, instruction_flags, d, a, b, excp_flag_user);
}

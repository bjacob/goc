// SPDX-License-Identifier: MIT

#include "rdna4_carry.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_dpp.h"

#include <stdint.h>

namespace {

template <goc::CarryOp Op, bool WithCarry>
int run(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d, uint32_t *carry,
        const uint32_t *const *a, const uint32_t *const *b, uint32_t input_carry) {
  if ((mode >> 32) && (!(mode & (GOC_DPP8 | GOC_DPP16)) || !goc::valid_dpp(mode)))
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, uint32_t(mode) & ~GOC_ALU_CLAMP, true))
    return error;
  uint32_t mask = exec_mask;
  if (!mask) {
    *carry = 0;
    return GOC_SUCCESS;
  }
  if (mode >> 32) {
    uint32_t permuted[32];
    const uint32_t *source = permuted;
    const uint32_t *input = Op == goc::CarryOp::Subrev ? b[0] : a[0];
    if (mode & GOC_DPP8)
      goc::dpp8_source(flags, mask, mode, permuted, input);
    else
      mask = goc::dpp16_source(flags, mask, mode, permuted, input);
    return run<Op, WithCarry>(flags, mask, uint32_t(mode), d, carry,
                              Op == goc::CarryOp::Subrev ? a : &source,
                              Op == goc::CarryOp::Subrev ? &source : b, input_carry);
  }
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    *carry = goc::carry_x86_64_v4<Op, WithCarry>(mask, mode, d, a, b, input_carry);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    *carry = goc::carry_x86_64_v3<Op, WithCarry>(mask, mode, d, a, b, input_carry);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[32], output_carry = 0;
  for (unsigned lane = 0; lane < 32; ++lane) {
    int64_t av = a[0][lane], bv = b[0][lane];
    uint32_t ci = WithCarry ? ((input_carry >> lane) & 1) : 0;
    int64_t wide;
    if constexpr (Op == goc::CarryOp::Add)
      wide = av + bv + ci;
    else if constexpr (Op == goc::CarryOp::Sub)
      wide = av - bv - ci;
    else
      wide = bv - av - ci;
    bool co = wide < 0 || wide > UINT32_MAX;
    result[lane] =
        (mode & GOC_ALU_CLAMP) && co ? (Op == goc::CarryOp::Add ? UINT32_MAX : 0) : uint32_t(wide);
    output_carry |= uint32_t(co) << lane;
  }
  for (unsigned lane = 0; lane < 32; ++lane)
    if ((mask >> lane) & 1)
      d[0][lane] = result[lane];
  *carry = output_carry & mask;
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_add_co_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, uint32_t *carry, const uint32_t *const *a,
                           const uint32_t *const *b) {
  return run<goc::CarryOp::Add, false>(flags, exec_mask, instruction_flags, d, carry, a, b, 0);
}

int goc_rdna4_v_sub_co_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, uint32_t *carry, const uint32_t *const *a,
                           const uint32_t *const *b) {
  return run<goc::CarryOp::Sub, false>(flags, exec_mask, instruction_flags, d, carry, a, b, 0);
}

int goc_rdna4_v_subrev_co_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, uint32_t *carry, const uint32_t *const *a,
                              const uint32_t *const *b) {
  return run<goc::CarryOp::Subrev, false>(flags, exec_mask, instruction_flags, d, carry, a, b, 0);
}

int goc_rdna4_v_add_co_ci_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, uint32_t *carry, const uint32_t *const *a,
                              const uint32_t *const *b, uint32_t input_carry) {
  return run<goc::CarryOp::Add, true>(flags, exec_mask, instruction_flags, d, carry, a, b,
                                      input_carry);
}

int goc_rdna4_v_sub_co_ci_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, uint32_t *carry, const uint32_t *const *a,
                              const uint32_t *const *b, uint32_t input_carry) {
  return run<goc::CarryOp::Sub, true>(flags, exec_mask, instruction_flags, d, carry, a, b,
                                      input_carry);
}

int goc_rdna4_v_subrev_co_ci_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, uint32_t *carry, const uint32_t *const *a,
                                 const uint32_t *const *b, uint32_t input_carry) {
  return run<goc::CarryOp::Subrev, true>(flags, exec_mask, instruction_flags, d, carry, a, b,
                                         input_carry);
}

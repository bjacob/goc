// SPDX-License-Identifier: MIT

#include "rdna4_unary.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_alu.h"

#include <stdint.h>

namespace {

template <goc::Unary Op>
int unary(uint64_t flags, uint64_t mask, uint32_t modifiers, uint32_t *const *d,
          const uint32_t *const *a) {
  const uint32_t known = GOC_ALU_NEG_A | GOC_ALU_ABS_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP;
  if (int error = goc::validate(flags, modifiers & ~known))
    return error;
  if (uint32_t(mask) == 0)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V3)
  if constexpr (Op != goc::Unary::Exp && Op != goc::Unary::Log) {
    if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
      goc::unary_x86_64_v3(Op, uint32_t(mask), modifiers, d[0], a[0]);
      return GOC_SUCCESS;
    }
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane)
    result[lane] = goc::as_bits(
        goc::alu_output(goc::unary_value<Op>(goc::alu_input(a[0][lane], modifiers)), modifiers));
  for (int lane = 0; lane < 32; ++lane)
    if ((mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_trunc_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                          uint32_t *const *d, const uint32_t *const *a) {
  return unary<goc::Unary::Trunc>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_ceil_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a) {
  return unary<goc::Unary::Ceil>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_rndne_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                          uint32_t *const *d, const uint32_t *const *a) {
  return unary<goc::Unary::Rndne>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_floor_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                          uint32_t *const *d, const uint32_t *const *a) {
  return unary<goc::Unary::Floor>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_sqrt_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a) {
  return unary<goc::Unary::Sqrt>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_rcp_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a) {
  return unary<goc::Unary::Rcp>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_rsq_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a) {
  return unary<goc::Unary::Rsq>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_exp_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a) {
  return unary<goc::Unary::Exp>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_log_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a) {
  return unary<goc::Unary::Log>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_fract_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                          uint32_t *const *d, const uint32_t *const *a) {
  return unary<goc::Unary::Fract>(flags, exec_mask, instruction_flags, d, a);
}

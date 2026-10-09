// SPDX-License-Identifier: MIT
// FP32 transcendental flushing follows rocjitsu's shared execution models
// and is checked against gfx1201 hardware captures.

#include "rdna4_unary.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_alu.h"
#include "rdna4_dpp.h"

#include <stdint.h>

namespace {

template <goc::Unary Op>
int unary(uint64_t flags, uint32_t exec_mask, uint64_t modifiers, uint32_t *const *d,
          const uint32_t *const *a) {
  if (modifiers >> 32) {
    return goc::execute_dpp(flags, exec_mask, modifiers, a,
                            [&](uint32_t exec_mask, const uint32_t *const *source) {
                              return unary<Op>(flags, exec_mask, uint32_t(modifiers), d, source);
                            });
  }

  const uint32_t known = GOC_ALU_NEG_A | GOC_ALU_ABS_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP;
  if (int error = goc::validate(flags, modifiers & ~known))
    return error;
  if (exec_mask == 0)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V3)
  if constexpr (Op != goc::Unary::Exp && Op != goc::Unary::Log) {
    if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
      goc::unary_x86_64_v3(Op, exec_mask, modifiers, d[0], a[0]);
      return GOC_SUCCESS;
    }
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    float value = goc::alu_input(a[0][lane], modifiers);
    if constexpr (goc::unary_flushes_f32<Op>)
      value = goc::flush_denorm_f32(value);
    value = goc::unary_value<Op>(value);
    if constexpr (goc::unary_flushes_f32<Op>)
      value = goc::flush_denorm_f32(value);
    result[lane] = goc::as_bits(goc::alu_output_f32(value, modifiers));
  }
  for (int lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_trunc_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                          uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::Trunc>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_ceil_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::Ceil>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_rndne_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                          uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::Rndne>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_floor_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                          uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::Floor>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_sqrt_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::Sqrt>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_rcp_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::Rcp>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_rsq_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::Rsq>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_exp_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::Exp>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_log_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::Log>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_fract_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                          uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::Fract>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_frexp_mant_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::FrexpMant>(flags, exec_mask, mode, d, a);
}

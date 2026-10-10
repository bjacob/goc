// SPDX-License-Identifier: MIT
// FP32 transcendental flushing follows rocjitsu's shared execution models
// and is checked against gfx1201 hardware captures.

#include "unary.h"
#include "alu.h"
#include "dpp.h"
#include "exp_model.h"
#include "goc/goc.h"
#include "internal.h"
#include "log_model.h"
#include "rcp_model.h"
#include "rsq_model.h"
#include "sqrt_model.h"

#include <stdint.h>

namespace {

// Exact captured unary mappings; all arithmetic here is on integer bits.
template <goc::Unary Op>
[[gnu::noinline]] int unary_exact(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a, bool rdna3) {
  if (mode >> 32)
    return goc::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return unary_exact<Op>(flags, exec_mask, uint32_t(mode), d, source, rdna3);
        });
  const uint32_t known = GOC_ALU_NEG_A | GOC_ALU_ABS_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP;
  if (int error = goc::validate(flags, mode & ~known, true,
                                GOC_FP_IEEE_MODE | GOC_FP_FLUSH_INPUT_DENORMALS |
                                    GOC_FP_FLUSH_OUTPUT_DENORMALS))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
  if (rdna3 && (flags & GOC_FP_IEEE_MODE))
    mode &= ~uint64_t(GOC_ALU_OMOD_HALF);
  unsigned omod = (mode >> 6) & 3;
  uint32_t result[32];
  for (unsigned lane = 0; lane < 32; ++lane) {
    uint32_t input = a[0][lane];
    if (mode & GOC_ALU_ABS_A)
      input &= 0x7fffffff;
    if (mode & GOC_ALU_NEG_A)
      input ^= 0x80000000;
    bool quiet_snan = !rdna3 || (flags & GOC_FP_IEEE_MODE);
    uint32_t value;
    if constexpr (Op == goc::Unary::Rcp)
      value = goc::rcp_f32_bits(input, quiet_snan);
    else if constexpr (Op == goc::Unary::Sqrt)
      value = goc::sqrt_f32_bits(input, quiet_snan);
    else if constexpr (Op == goc::Unary::Rsq)
      value = goc::rsq_f32_bits(input, quiet_snan);
    else if constexpr (Op == goc::Unary::Exp)
      value = goc::exp_f32_bits(input, quiet_snan);
    else
      value = goc::log_f32_bits(input, quiet_snan);
    if (!(value & 0x7f800000))
      value &= 0x80000000;
    uint32_t magnitude = value & 0x7fffffff;
    if (omod && magnitude < 0x7f800000) {
      unsigned exponent = magnitude >> 23;
      if (!exponent)
        value = 0;
      else if (omod == 3 && exponent == 1)
        value &= 0x80000000;
      else if (omod != 3 && exponent + omod >= 255)
        value = (value & 0x80000000) | 0x7f800000;
      else
        value = omod == 3 ? value - 0x800000 : value + (omod << 23);
    }
    if (mode & GOC_ALU_CLAMP)
      value = (value & 0x80000000) || (value & 0x7fffffff) > 0x7f800000 ? 0
              : value > 0x3f800000                                      ? 0x3f800000
                                                                        : value;
    result[lane] = value;
  }
  for (unsigned lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

template <goc::Unary Op, bool Rdna3 = false>
int unary(uint64_t flags, uint32_t exec_mask, uint64_t modifiers, uint32_t *const *d,
          const uint32_t *const *a) {
  if constexpr (Op == goc::Unary::Rcp || Op == goc::Unary::Sqrt || Op == goc::Unary::Rsq ||
                Op == goc::Unary::Exp || Op == goc::Unary::Log)
    if ((flags & GOC_SEMANTICS_MASK) == GOC_SEMANTICS_EXACT_EMPIRICAL)
      return unary_exact<Op>(flags, exec_mask, modifiers, d, a, Rdna3);
  if constexpr (Rdna3) {
    // These transcendental operations use forced-flush OMOD policy, regardless
    // of guest denormal flags. Only RDNA3 IEEE mode disables output scaling.
    if (flags & GOC_FP_IEEE_MODE)
      modifiers &= ~uint64_t(GOC_ALU_OMOD_HALF);
    return unary<Op>(flags, exec_mask, modifiers, d, a);
  }
  if (modifiers >> 32) {
    return goc::execute_dpp(
        flags, exec_mask, modifiers, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return unary<Op, Rdna3>(flags, exec_mask, uint32_t(modifiers), d, source);
        });
  }

  const uint32_t known = GOC_ALU_NEG_A | GOC_ALU_ABS_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP;
  constexpr uint64_t fp_flags = goc::unary_flushes_f32<Op> ? GOC_FP_FLUSH_INPUT_DENORMALS |
                                                                 GOC_FP_FLUSH_OUTPUT_DENORMALS |
                                                                 GOC_FP_IEEE_MODE
                                                           : 0;
  if (int error = goc::validate(flags, modifiers & ~known, false, fp_flags))
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

template <goc::Unary Op, bool Rdna3 = false>
int unary_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                 const uint32_t *const *a) {
  if (int error = unary<Op, Rdna3>(flags, 0, mode, nullptr, nullptr))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
  for (unsigned half = 0; half < 2; ++half) {
    uint64_t half_mode = mode;
    if (mode & GOC_DPP16) {
      uint64_t rows = (mode & GOC_DPP_ROW_MASK) >> GOC_DPP_ROW_SHIFT;
      half_mode = (mode & ~GOC_DPP_ROW_MASK) | (((rows >> (half * 2)) & 3) << GOC_DPP_ROW_SHIFT);
    }
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *source = a[0] + half * 32;
    if (int error =
            unary<Op, Rdna3>(flags, uint32_t(exec_mask >> (half * 32)), half_mode, &dest, &source))
      return error;
  }
  return GOC_SUCCESS;
}

} // namespace

int goc_v_trunc_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                    uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::Trunc>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_ceil_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                   uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::Ceil>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_rndne_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                    uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::Rndne>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_floor_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                    uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::Floor>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_sqrt_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                   uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::Sqrt, true>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_sqrt_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::Sqrt>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_rcp_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::Rcp, true>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_rcp_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::Rcp>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_rsq_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::Rsq, true>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_rsq_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::Rsq>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_exp_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::Exp, true>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_exp_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::Exp>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_log_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::Log, true>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_log_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::Log>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_fract_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                    uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::Fract>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_frexp_mant_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::FrexpMant>(flags, exec_mask, mode, d, a);
}

int goc_v_trunc_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary_wave64<goc::Unary::Trunc>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_ceil_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                          uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary_wave64<goc::Unary::Ceil>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_rndne_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary_wave64<goc::Unary::Rndne>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_floor_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary_wave64<goc::Unary::Floor>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_sqrt_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                          uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary_wave64<goc::Unary::Sqrt, true>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_sqrt_f32_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary_wave64<goc::Unary::Sqrt>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_rcp_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary_wave64<goc::Unary::Rcp, true>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_rcp_f32_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary_wave64<goc::Unary::Rcp>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_rsq_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary_wave64<goc::Unary::Rsq, true>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_rsq_f32_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary_wave64<goc::Unary::Rsq>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_exp_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary_wave64<goc::Unary::Exp, true>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_exp_f32_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary_wave64<goc::Unary::Exp>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_log_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary_wave64<goc::Unary::Log, true>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_log_f32_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary_wave64<goc::Unary::Log>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_fract_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary_wave64<goc::Unary::Fract>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_frexp_mant_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a,
                                uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary_wave64<goc::Unary::FrexpMant>(flags, exec_mask, mode, d, a);
}

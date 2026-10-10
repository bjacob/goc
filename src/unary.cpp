// SPDX-License-Identifier: MIT
// FP32 transcendental flushing follows rocjitsu's shared execution models
// and is checked against gfx1201 hardware captures.

#include "unary.h"
#include "alu.h"
#include "dpp.h"
#include "goc/goc.h"
#include "internal.h"
#include "unary_exact.h"

#include <stdint.h>

namespace {

// Exact unary mappings adapted from rocjitsu; all arithmetic here is on integer bits.
template <goc::Unary Op>
[[gnu::noinline]] int unary_exact(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a, bool rdna3) {
  if (mode >> 32)
    return goc::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return unary_exact<Op>(flags, exec_mask, uint32_t(mode), d, source, rdna3);
        });
  const uint32_t known = GOC_ALU_NEG_A | GOC_ALU_ABS_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP;
  if (int error =
          goc::validate(flags, mode & ~known, true,
                        GOC_FP_IEEE_MODE | GOC_FP_FLUSH_INPUT_DENORMALS |
                            GOC_FP_FLUSH_OUTPUT_DENORMALS | (rdna3 ? GOC_FP_DX10_CLAMP_OFF : 0)))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
  if (rdna3 && ((flags & GOC_FP_IEEE_MODE) ||
                (!goc::unary_flushes_f32<Op> && !(flags & GOC_FP_FLUSH_OUTPUT_DENORMALS))))
    mode &= ~uint64_t(GOC_ALU_OMOD_HALF);
  uint32_t result[32];
  for (unsigned lane = 0; lane < 32; ++lane) {
    result[lane] = goc::unary_exact_value<Op>(flags, uint32_t(mode), a[0][lane], rdna3);
  }
  for (unsigned lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

// Separate the two scalar clamp policies at compile time so the default loop
// retains its existing vectorization. Both architectures share these kernels.
template <goc::Unary Op, bool PreserveClampNan = false>
int unary_loose(uint64_t flags, uint32_t exec_mask, uint64_t modifiers, uint32_t *const *d,
                const uint32_t *const *a) {
  if (modifiers >> 32) {
    return goc::execute_dpp(flags, exec_mask, modifiers, a,
                            [&](uint32_t exec_mask, const uint32_t *const *source) {
                              return unary_loose<Op, PreserveClampNan>(
                                  flags, exec_mask, uint32_t(modifiers), d, source);
                            });
  }

  const uint32_t known = GOC_ALU_NEG_A | GOC_ALU_ABS_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP;
  constexpr uint64_t fp_flags = GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS |
                                GOC_FP_IEEE_MODE | (PreserveClampNan ? GOC_FP_DX10_CLAMP_OFF : 0);
  if (int error = goc::validate(flags, modifiers & ~known, false, fp_flags))
    return error;
  if (exec_mask == 0)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::unary_x86_64_v3(Op, exec_mask, modifiers, d[0], a[0], flags);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    float value = goc::alu_input(a[0][lane], modifiers);
    if constexpr (goc::unary_flushes_f32<Op>)
      value = goc::flush_denorm_f32(value);
    else {
      if (flags & GOC_FP_FLUSH_INPUT_DENORMALS)
        value = goc::flush_denorm_f32(value);
    }
    value = goc::unary_value<Op>(value);
    if constexpr (goc::unary_flushes_f32<Op>)
      value = goc::flush_denorm_f32(value);
    else if constexpr (Op == goc::Unary::Fract) {
      if ((flags & GOC_FP_FLUSH_OUTPUT_DENORMALS) && !(modifiers & GOC_ALU_OMOD_HALF))
        value = goc::flush_denorm_f32(value);
    }
    uint32_t output_mode = modifiers;
    if constexpr (PreserveClampNan)
      if ((goc::as_bits(value) & 0x7fffffff) > 0x7f800000)
        output_mode &= ~GOC_ALU_CLAMP;
    result[lane] = goc::as_bits(goc::alu_output_f32(value, output_mode));
  }
  for (int lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

template <goc::Unary Op, bool Rdna3 = false>
int unary(uint64_t flags, uint32_t exec_mask, uint64_t modifiers, uint32_t *const *d,
          const uint32_t *const *a, uint32_t *excp_flag_user = nullptr) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if constexpr (Op != goc::Unary::Fract)
    if ((flags & GOC_SEMANTICS_MASK) == GOC_SEMANTICS_EXACT_EMPIRICAL)
      return unary_exact<Op>(flags, exec_mask, modifiers, d, a, Rdna3);
  if constexpr (Rdna3) {
    // Transcendentals use forced-flush OMOD policy. Other unary operations
    // use the guest output-denormal policy, even where their result cannot
    // be subnormal. Both policies also honor RDNA3 IEEE mode.
    if ((flags & GOC_FP_IEEE_MODE) ||
        (!goc::unary_flushes_f32<Op> && !(flags & GOC_FP_FLUSH_OUTPUT_DENORMALS)))
      modifiers &= ~uint64_t(GOC_ALU_OMOD_HALF);
    if (flags & GOC_FP_DX10_CLAMP_OFF)
      return unary_loose<Op, true>(flags, exec_mask, modifiers, d, a);
  }
  return unary_loose<Op>(flags, exec_mask, modifiers, d, a);
}

template <goc::Unary Op, bool Rdna3 = false>
int unary_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                 const uint32_t *const *a, uint32_t *excp_flag_user = nullptr) {
  if (int error = unary<Op, Rdna3>(flags, 0, mode, nullptr, nullptr, excp_flag_user))
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
    if (int error = unary<Op, Rdna3>(flags, uint32_t(exec_mask >> (half * 32)), half_mode, &dest,
                                     &source, excp_flag_user))
      return error;
  }
  return GOC_SUCCESS;
}

} // namespace

int goc_v_trunc_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                    uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::Trunc, true>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_trunc_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                          uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::Trunc>(flags, exec_mask, instruction_flags, d, a, excp_flag_user);
}

int goc_v_ceil_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                   uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::Ceil, true>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_ceil_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::Ceil>(flags, exec_mask, instruction_flags, d, a, excp_flag_user);
}

int goc_v_rndne_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                    uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::Rndne, true>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_rndne_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                          uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::Rndne>(flags, exec_mask, instruction_flags, d, a, excp_flag_user);
}

int goc_v_floor_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                    uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::Floor, true>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_floor_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                          uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::Floor>(flags, exec_mask, instruction_flags, d, a, excp_flag_user);
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
  return unary<goc::Unary::Sqrt>(flags, exec_mask, instruction_flags, d, a, excp_flag_user);
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
  return unary<goc::Unary::Rcp>(flags, exec_mask, instruction_flags, d, a, excp_flag_user);
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
  return unary<goc::Unary::Rsq>(flags, exec_mask, instruction_flags, d, a, excp_flag_user);
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
  return unary<goc::Unary::Exp>(flags, exec_mask, instruction_flags, d, a, excp_flag_user);
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
  return unary<goc::Unary::Log>(flags, exec_mask, instruction_flags, d, a, excp_flag_user);
}

int goc_v_fract_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                    uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::Fract, true>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_fract_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                          uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::Fract>(flags, exec_mask, instruction_flags, d, a, excp_flag_user);
}

int goc_v_frexp_mant_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::FrexpMant, true>(flags, exec_mask, mode, d, a);
}

int goc_v_frexp_mant_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary<goc::Unary::FrexpMant>(flags, exec_mask, mode, d, a, excp_flag_user);
}

int goc_v_trunc_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary_wave64<goc::Unary::Trunc, true>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_trunc_f32_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary_wave64<goc::Unary::Trunc>(flags, exec_mask, instruction_flags, d, a, excp_flag_user);
}

int goc_v_ceil_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                          uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary_wave64<goc::Unary::Ceil, true>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_ceil_f32_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary_wave64<goc::Unary::Ceil>(flags, exec_mask, instruction_flags, d, a, excp_flag_user);
}

int goc_v_rndne_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary_wave64<goc::Unary::Rndne, true>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_rndne_f32_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary_wave64<goc::Unary::Rndne>(flags, exec_mask, instruction_flags, d, a, excp_flag_user);
}

int goc_v_floor_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary_wave64<goc::Unary::Floor, true>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_floor_f32_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary_wave64<goc::Unary::Floor>(flags, exec_mask, instruction_flags, d, a, excp_flag_user);
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
  return unary_wave64<goc::Unary::Sqrt>(flags, exec_mask, instruction_flags, d, a, excp_flag_user);
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
  return unary_wave64<goc::Unary::Rcp>(flags, exec_mask, instruction_flags, d, a, excp_flag_user);
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
  return unary_wave64<goc::Unary::Rsq>(flags, exec_mask, instruction_flags, d, a, excp_flag_user);
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
  return unary_wave64<goc::Unary::Exp>(flags, exec_mask, instruction_flags, d, a, excp_flag_user);
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
  return unary_wave64<goc::Unary::Log>(flags, exec_mask, instruction_flags, d, a, excp_flag_user);
}

int goc_v_fract_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary_wave64<goc::Unary::Fract, true>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_fract_f32_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary_wave64<goc::Unary::Fract>(flags, exec_mask, instruction_flags, d, a, excp_flag_user);
}

int goc_v_frexp_mant_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary_wave64<goc::Unary::FrexpMant, true>(flags, exec_mask, mode, d, a);
}

int goc_v_frexp_mant_f32_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                      uint32_t *const *d, const uint32_t *const *a,
                                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return unary_wave64<goc::Unary::FrexpMant>(flags, exec_mask, mode, d, a, excp_flag_user);
}

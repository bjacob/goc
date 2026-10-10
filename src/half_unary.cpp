// SPDX-License-Identifier: MIT

#include "half_unary.h"
#include "alu.h"
#include "bits.h"
#include "dpp.h"
#include "float_formats.h"
#include "goc/goc.h"
#include "internal.h"
#include "rcp_model.h"
#include "round_integral.h"
#include "rsq_model.h"
#include "sqrt_model.h"
#include "unary.h"

#include <algorithm>
#include <cmath>
#include <stdint.h>

namespace {

template <goc::Unary Op>
constexpr bool supports_exact =
    Op == goc::Unary::Trunc || Op == goc::Unary::Ceil || Op == goc::Unary::Rndne ||
    Op == goc::Unary::Floor || Op == goc::Unary::FrexpMant || goc::half_unary_sfu<Op>;

static const uint64_t fp_flags =
    GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS | GOC_FP_IEEE_MODE;

template <goc::Unary Op, bool Rdna3 = false>
[[gnu::noinline]] int unary_exact(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a) {
  if (mode >> 32)
    return goc::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return unary_exact<Op, Rdna3>(flags, exec_mask, uint32_t(mode), d, source);
        });
  const uint32_t known = GOC_ALU_NEG_A | GOC_ALU_ABS_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP |
                         GOC_ALU_HIGH_A | GOC_ALU_HIGH_D;
  if (int error = goc::validate(flags, mode & ~known, true, fp_flags))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
  unsigned a_shift = mode & GOC_ALU_HIGH_A ? 16 : 0;
  unsigned d_shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  unsigned omod = (mode >> 6) & 3;
  uint16_t result[32];
  for (unsigned lane = 0; lane < 32; ++lane) {
    uint32_t input = uint16_t(a[0][lane] >> a_shift);
    if (mode & GOC_ALU_ABS_A)
      input &= 0x7fff;
    if (mode & GOC_ALU_NEG_A)
      input ^= 0x8000;
    uint32_t magnitude = input & 0x7fff;

    uint32_t value;
    if constexpr (goc::half_unary_sfu<Op>) {
      // rocjitsu transcendental::map_f16: apply the half input policy, evaluate
      // the borrowed integer FP32 model, then narrow before output modifiers.
      if ((flags & GOC_FP_FLUSH_INPUT_DENORMALS) && magnitude < 0x400)
        input &= 0x8000;
      uint32_t wide_input = goc::as_bits(goc::f16_to_float(uint16_t(input)));
      bool quiet_snan = !Rdna3 || (flags & GOC_FP_IEEE_MODE);
      uint32_t wide_result;
      if constexpr (Op == goc::Unary::Rcp)
        wide_result = goc::rcp_f32_bits(wide_input, quiet_snan);
      else if constexpr (Op == goc::Unary::Rsq)
        wide_result = goc::rsq_f32_bits(wide_input, quiet_snan);
      else
        wide_result = goc::sqrt_f32_bits(wide_input, quiet_snan);
      value = goc::float_to_f16(goc::as_float(wide_result), flags & GOC_FP16_OVFL);
      if ((flags & GOC_FP16_OVFL) && (wide_result & 0x7fffffff) == 0x7f800000 && magnitude < 0x7c00)
        value = (value & 0x8000) | 0x7bff;

      if ((flags & GOC_FP_FLUSH_OUTPUT_DENORMALS) && (value & 0x7fff) < 0x400)
        value &= 0x8000;
    } else if constexpr (Op == goc::Unary::FrexpMant) {
      // Same integer normalization as rocjitsu's FREXP, with half-width fields.
      if (magnitude >= 0x7c00)
        value = input | (magnitude > 0x7c00 ? 0x200 : 0);
      else if (!magnitude || (magnitude < 0x400 && (flags & GOC_FP_FLUSH_INPUT_DENORMALS)))
        value = input & 0x8000;
      else {
        unsigned shift = magnitude < 0x400 ? goc::bit_count_zero<false>(magnitude) - 21 : 0;
        value = (input & 0x8000) | 0x3800 | ((magnitude << shift) & 0x3ff);
      }
    } else {
      constexpr auto rounding = Op == goc::Unary::Trunc   ? goc::IntegralRound::Trunc
                                : Op == goc::Unary::Ceil  ? goc::IntegralRound::Ceil
                                : Op == goc::Unary::Floor ? goc::IntegralRound::Floor
                                                          : goc::IntegralRound::Nearest;
      value = goc::round_integral_bits<rounding, true>(input, flags & GOC_FP_FLUSH_INPUT_DENORMALS);
    }
    // OMOD flushes tiny inputs to +0; newly tiny div:2 results retain their sign.
    magnitude = value & 0x7fff;
    if (omod && magnitude < 0x7c00) {
      if (!magnitude || ((Op == goc::Unary::Rcp) && magnitude < 0x400))
        value = 0;
      else if (omod == 3 && (Op == goc::Unary::Rcp) && magnitude < 0x800)
        value &= 0x8000;
      else if (omod == 3)
        value -= 0x400;
      else if ((magnitude >> 10) + omod >= 31) {
        value = (value & 0x8000) | (flags & GOC_FP16_OVFL ? 0x7bff : 0x7c00);
      } else
        value += omod << 10;
    }
    if (mode & GOC_ALU_CLAMP)
      value = (value & 0x8000) || (value & 0x7fff) > 0x7c00 ? 0 : value > 0x3c00 ? 0x3c00 : value;
    result[lane] = uint16_t(value);
  }
  for (unsigned lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = (d[0][lane] & ~(65535U << d_shift)) | (uint32_t(result[lane]) << d_shift);
  return GOC_SUCCESS;
}

template <goc::Unary Op>
int unary(uint64_t flags, uint32_t exec_mask, uint64_t modifiers, uint32_t *const *d,
          const uint32_t *const *a) {
  if (modifiers >> 32)
    return goc::execute_dpp(flags, exec_mask, modifiers, a,
                            [&](uint32_t exec_mask, const uint32_t *const *source) {
                              return unary<Op>(flags, exec_mask, uint32_t(modifiers), d, source);
                            });
  const uint32_t known = GOC_ALU_NEG_A | GOC_ALU_ABS_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP |
                         GOC_ALU_HIGH_A | GOC_ALU_HIGH_D;
  if (int error = goc::validate(flags, modifiers & ~known, false,
                                (supports_exact<Op> || Op == goc::Unary::Fract) ? fp_flags : 0))
    return error;
  if (exec_mask == 0)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::half_unary_x86_64_v3<Op>(flags, exec_mask, modifiers, d[0], a[0]);
    return GOC_SUCCESS;
  }
#endif
  if constexpr (supports_exact<Op> && !goc::half_unary_sfu<Op>)
    return unary_exact<Op>(flags, exec_mask, modifiers, d, a);
  int a_shift = modifiers & GOC_ALU_HIGH_A ? 16 : 0;
  int d_shift = modifiers & GOC_ALU_HIGH_D ? 16 : 0;
  uint16_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    uint16_t input = uint16_t(a[0][lane] >> a_shift);
    if constexpr (goc::half_unary_sfu<Op> || Op == goc::Unary::Fract)
      if ((flags & GOC_FP_FLUSH_INPUT_DENORMALS) && (input & 0x7fff) < 0x400)
        input &= 0x8000;
    float x = goc::alu_input(goc::as_bits(goc::f16_to_float(input)), modifiers);
    if constexpr (Op == goc::Unary::Exp) {
      // Bound finite intermediates without changing any final FP16 result,
      // including finite-overflow saturation and output scaling by up to four.
      if (std::isfinite(x))
        x = std::clamp(x, -64.0f, 64.0f);
    }
    float value = goc::unary_value<Op>(x);
    if constexpr (Op == goc::Unary::Log) {
      if (x == 0 && (flags & GOC_FP16_OVFL))
        value = -65504;
    }
    if constexpr (Op == goc::Unary::Rcp || Op == goc::Unary::Rsq)
      if ((flags & GOC_FP16_OVFL) && x == 0)
        value = std::copysign(65504.0f, value);
    if constexpr (Op == goc::Unary::Exp || Op == goc::Unary::Log || goc::half_unary_sfu<Op>) {
      // Power-of-two scaling commutes with normal half rounding. For these
      // three SFUs, CLAMP also removes the overflow distinction, so loose mode
      // can omit the intermediate conversion when CLAMP is present.
      if ((modifiers & GOC_ALU_OMOD_HALF) &&
          (!goc::half_unary_sfu<Op> || !(modifiers & GOC_ALU_CLAMP)))
        value = goc::f16_to_float(goc::float_to_f16(value, flags & GOC_FP16_OVFL));
    }
    if constexpr (Op == goc::Unary::Fract) {
      // Keep a fraction strictly below one after FP16 rounding.
      if (value > 0x1.ffcp-1f)
        value = 0x1.ffcp-1f;
    }
    if constexpr (goc::half_unary_sfu<Op> || Op == goc::Unary::Fract)
      if ((flags & GOC_FP_FLUSH_OUTPUT_DENORMALS) && std::abs(value) < 0x1p-14f)
        value = std::copysign(0.0f, value);
    result[lane] = goc::float_to_f16(goc::alu_output_f16(value, modifiers), flags & GOC_FP16_OVFL);
  }
  for (int lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = (d[0][lane] & ~(0xffffU << d_shift)) | (uint32_t(result[lane]) << d_shift);
  return GOC_SUCCESS;
}

template <goc::Unary Op, bool Rdna3 = false>
int run(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
        const uint32_t *const *a, uint32_t *state) {
  if (state && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if constexpr (Rdna3)
    if ((flags & GOC_FP_IEEE_MODE) || !(flags & GOC_FP_FLUSH_OUTPUT_DENORMALS))
      mode &= ~uint64_t(GOC_ALU_OMOD_HALF);
  if constexpr (supports_exact<Op>) {
    if ((flags & GOC_SEMANTICS_MASK) == GOC_SEMANTICS_EXACT_EMPIRICAL)
      return unary_exact<Op, (Rdna3 && goc::half_unary_sfu<Op>)>(flags, exec_mask, mode, d, a);
  }
  return unary<Op>(flags, exec_mask, mode, d, a);
}

template <goc::Unary Op, bool Rdna3 = false>
int unary_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                 const uint32_t *const *a, uint32_t *state) {
  if (int error = run<Op, Rdna3>(flags, 0, mode, nullptr, nullptr, state))
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
    if (int error = run<Op, Rdna3>(flags, uint32_t(exec_mask >> (half * 32)), half_mode, &dest,
                                   &source, state))
      return error;
  }
  return GOC_SUCCESS;
}

} // namespace

int goc_v_trunc_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                    uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return run<goc::Unary::Trunc, true>(flags, exec_mask, instruction_flags, d, a, trapsts);
}

int goc_v_trunc_f16_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                          uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  return run<goc::Unary::Trunc, false>(flags, exec_mask, instruction_flags, d, a, excp_flag_user);
}

int goc_v_ceil_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                   uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return run<goc::Unary::Ceil, true>(flags, exec_mask, instruction_flags, d, a, trapsts);
}

int goc_v_ceil_f16_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  return run<goc::Unary::Ceil, false>(flags, exec_mask, instruction_flags, d, a, excp_flag_user);
}

int goc_v_rndne_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                    uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return run<goc::Unary::Rndne, true>(flags, exec_mask, instruction_flags, d, a, trapsts);
}

int goc_v_rndne_f16_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                          uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  return run<goc::Unary::Rndne, false>(flags, exec_mask, instruction_flags, d, a, excp_flag_user);
}

int goc_v_floor_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                    uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return run<goc::Unary::Floor, true>(flags, exec_mask, instruction_flags, d, a, trapsts);
}

int goc_v_floor_f16_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                          uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  return run<goc::Unary::Floor, false>(flags, exec_mask, instruction_flags, d, a, excp_flag_user);
}

int goc_v_sqrt_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                   uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return run<goc::Unary::Sqrt, true>(flags, exec_mask, instruction_flags, d, a, trapsts);
}

int goc_v_sqrt_f16_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  return run<goc::Unary::Sqrt, false>(flags, exec_mask, instruction_flags, d, a, excp_flag_user);
}

int goc_v_rcp_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return run<goc::Unary::Rcp, true>(flags, exec_mask, instruction_flags, d, a, trapsts);
}

int goc_v_rcp_f16_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  return run<goc::Unary::Rcp, false>(flags, exec_mask, instruction_flags, d, a, excp_flag_user);
}

int goc_v_rsq_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return run<goc::Unary::Rsq, true>(flags, exec_mask, instruction_flags, d, a, trapsts);
}

int goc_v_rsq_f16_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  return run<goc::Unary::Rsq, false>(flags, exec_mask, instruction_flags, d, a, excp_flag_user);
}

int goc_v_exp_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  return run<goc::Unary::Exp, false>(flags, exec_mask, instruction_flags, d, a, excp_flag_user);
}

int goc_v_log_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  return run<goc::Unary::Log, false>(flags, exec_mask, instruction_flags, d, a, excp_flag_user);
}

int goc_v_fract_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                    uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return run<goc::Unary::Fract, true>(flags, exec_mask, instruction_flags, d, a, trapsts);
}

int goc_v_fract_f16_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                          uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  return run<goc::Unary::Fract, false>(flags, exec_mask, instruction_flags, d, a, excp_flag_user);
}

int goc_v_frexp_mant_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return run<goc::Unary::FrexpMant, true>(flags, exec_mask, instruction_flags, d, a, trapsts);
}

int goc_v_frexp_mant_f16_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               uint32_t *excp_flag_user) {
  return run<goc::Unary::FrexpMant, false>(flags, exec_mask, instruction_flags, d, a,
                                           excp_flag_user);
}

int goc_v_trunc_f16_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return unary_wave64<goc::Unary::Trunc, true>(flags, exec_mask, instruction_flags, d, a, trapsts);
}

int goc_v_trunc_f16_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  return unary_wave64<goc::Unary::Trunc, false>(flags, exec_mask, instruction_flags, d, a,
                                                excp_flag_user);
}

int goc_v_ceil_f16_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                          uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return unary_wave64<goc::Unary::Ceil, true>(flags, exec_mask, instruction_flags, d, a, trapsts);
}

int goc_v_ceil_f16_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                uint32_t *excp_flag_user) {
  return unary_wave64<goc::Unary::Ceil, false>(flags, exec_mask, instruction_flags, d, a,
                                               excp_flag_user);
}

int goc_v_rndne_f16_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return unary_wave64<goc::Unary::Rndne, true>(flags, exec_mask, instruction_flags, d, a, trapsts);
}

int goc_v_rndne_f16_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  return unary_wave64<goc::Unary::Rndne, false>(flags, exec_mask, instruction_flags, d, a,
                                                excp_flag_user);
}

int goc_v_floor_f16_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return unary_wave64<goc::Unary::Floor, true>(flags, exec_mask, instruction_flags, d, a, trapsts);
}

int goc_v_floor_f16_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  return unary_wave64<goc::Unary::Floor, false>(flags, exec_mask, instruction_flags, d, a,
                                                excp_flag_user);
}

int goc_v_sqrt_f16_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                          uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return unary_wave64<goc::Unary::Sqrt, true>(flags, exec_mask, instruction_flags, d, a, trapsts);
}

int goc_v_sqrt_f16_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                uint32_t *excp_flag_user) {
  return unary_wave64<goc::Unary::Sqrt, false>(flags, exec_mask, instruction_flags, d, a,
                                               excp_flag_user);
}

int goc_v_rcp_f16_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return unary_wave64<goc::Unary::Rcp, true>(flags, exec_mask, instruction_flags, d, a, trapsts);
}

int goc_v_rcp_f16_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               uint32_t *excp_flag_user) {
  return unary_wave64<goc::Unary::Rcp, false>(flags, exec_mask, instruction_flags, d, a,
                                              excp_flag_user);
}

int goc_v_rsq_f16_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return unary_wave64<goc::Unary::Rsq, true>(flags, exec_mask, instruction_flags, d, a, trapsts);
}

int goc_v_rsq_f16_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               uint32_t *excp_flag_user) {
  return unary_wave64<goc::Unary::Rsq, false>(flags, exec_mask, instruction_flags, d, a,
                                              excp_flag_user);
}

int goc_v_exp_f16_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  return unary_wave64<goc::Unary::Exp, false>(flags, exec_mask, instruction_flags, d, a,
                                              excp_flag_user);
}

int goc_v_log_f16_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  return unary_wave64<goc::Unary::Log, false>(flags, exec_mask, instruction_flags, d, a,
                                              excp_flag_user);
}

int goc_v_fract_f16_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return unary_wave64<goc::Unary::Fract, true>(flags, exec_mask, instruction_flags, d, a, trapsts);
}

int goc_v_fract_f16_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  return unary_wave64<goc::Unary::Fract, false>(flags, exec_mask, instruction_flags, d, a,
                                                excp_flag_user);
}

int goc_v_frexp_mant_f16_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return unary_wave64<goc::Unary::FrexpMant, true>(flags, exec_mask, instruction_flags, d, a,
                                                   trapsts);
}

int goc_v_frexp_mant_f16_rdna4_wave64(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, uint32_t *excp_flag_user) {
  return unary_wave64<goc::Unary::FrexpMant, false>(flags, exec_mask, instruction_flags, d, a,
                                                    excp_flag_user);
}

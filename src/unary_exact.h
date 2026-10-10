// SPDX-License-Identifier: MIT

#pragma once

#include "exp_model.h"
#include "goc/goc.h"
#include "log_model.h"
#include "rcp_model.h"
#include "round_integral.h"
#include "rsq_model.h"
#include "sqrt_model.h"
#include "unary.h"

#include <stdint.h>

namespace goc {

// Evaluate one raw FP32 operand using the borrowed rocjitsu mappings and apply
// source/output modifiers. The caller supplies the effective architecture OMOD.
// All calculation uses integer bits, independently of host FP controls.
template <Unary Op>
[[gnu::always_inline]] inline uint32_t unary_exact_value(uint64_t flags, uint32_t mode,
                                                         uint32_t input, bool rdna3) {
  unsigned omod = (mode >> 6) & 3;
  if (mode & GOC_ALU_ABS_A)
    input &= 0x7fffffff;
  if (mode & GOC_ALU_NEG_A)
    input ^= 0x80000000;
  bool quiet_snan = !rdna3 || (flags & GOC_FP_IEEE_MODE);
  uint32_t value;
  if constexpr (goc::unary_rounds_f32<Op>) {
    constexpr auto rounding = Op == goc::Unary::Trunc   ? goc::IntegralRound::Trunc
                              : Op == goc::Unary::Ceil  ? goc::IntegralRound::Ceil
                              : Op == goc::Unary::Floor ? goc::IntegralRound::Floor
                                                        : goc::IntegralRound::Nearest;
    value = goc::round_integral_bits<rounding, false>(input, flags & GOC_FP_FLUSH_INPUT_DENORMALS);
  } else if constexpr (Op == goc::Unary::FrexpMant)
    value = goc::frexp_mant_f32_bits(input, flags & GOC_FP_FLUSH_INPUT_DENORMALS);
  else if constexpr (Op == goc::Unary::Rcp)
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
    else if (omod != 3 && exponent + omod >= 255) {
      value = (value & 0x80000000) | 0x7f800000;
    } else
      value = omod == 3 ? value - 0x800000 : value + (omod << 23);
  }
  if ((mode & GOC_ALU_CLAMP) &&
      (!(flags & GOC_FP_DX10_CLAMP_OFF) || (value & 0x7fffffff) <= 0x7f800000))
    value = (value & 0x80000000) || (value & 0x7fffffff) > 0x7f800000 ? 0
            : value > 0x3f800000                                      ? 0x3f800000
                                                                      : value;
  return value;
}

} // namespace goc

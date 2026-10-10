// SPDX-License-Identifier: MIT

#pragma once

#include "bits.h"
#include "float_formats.h"
#include "goc/goc.h"
#include "half_unary.h"
#include "rcp_model.h"
#include "round_integral.h"
#include "rsq_model.h"
#include "sqrt_model.h"
#include "unary.h"

#include <stdint.h>

namespace goc {

// Evaluate a raw half operand with the borrowed rocjitsu mapping and modifiers.
// The caller selects the source half and supplies the effective architecture OMOD.
// Keep this calculation inline in vector lane loops.
template <Unary Op, bool Rdna3 = false, bool PreserveClampNan = false>
[[gnu::always_inline]] inline uint16_t half_unary_exact_value(uint64_t flags, uint32_t mode,
                                                              uint16_t raw) {
  uint32_t input = raw;
  unsigned omod = (mode >> 6) & 3;
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
    // rocjitsu widens half inputs before FREXP, so half subnormals survive
    // every guest denormal policy. Normalize directly in half-width fields.
    if (magnitude >= 0x7c00)
      value = input | (magnitude > 0x7c00 ? 0x200 : 0);
    else if (magnitude >= 0x400)
      value = (input & 0x83ff) | 0x3800;
    else if (magnitude) {
      unsigned shift = goc::bit_count_zero<false>(magnitude) - 21;
      value = (input & 0x8000) | 0x3800 | ((magnitude << shift) & 0x3ff);
    } else
      value = input;
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
  if ((mode & GOC_ALU_CLAMP) && (!PreserveClampNan || (value & 0x7fff) <= 0x7c00))
    value = (value & 0x8000) || (value & 0x7fff) > 0x7c00 ? 0 : value > 0x3c00 ? 0x3c00 : value;
  return uint16_t(value);
}

} // namespace goc

// SPDX-License-Identifier: MIT

// Integer rounding extends rocjitsu util::rndne_scalar to directed rounding.
// GFX1201 verifies all FP16 patterns and raw FP32 boundary/random values.

#include "goc/goc.h"
#include "internal.h"
#include "round_integral.h"

#include <stdint.h>

namespace {

using goc::IntegralRound;

template <IntegralRound Op, bool Half>
int run(uint64_t flags, uint32_t mode, uint32_t *d, uint32_t a, uint32_t *excp_flag_user) {
  if (int error = goc::validate(flags, mode, true,
                                GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS))
    return error;
  constexpr unsigned fraction = Half ? 10 : 23;
  constexpr uint32_t sign_bit = Half ? 0x8000 : 0x80000000, inf = Half ? 0x7c00 : 0x7f800000;
  if constexpr (Half)
    a &= 65535;
  // GFX1201: rounding instructions report input classification, not INEXACT
  // when the fractional part is discarded. Loose mode skips this work.
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) == GOC_SEMANTICS_EXACT_EMPIRICAL) {
    const uint32_t magnitude = a & (sign_bit - 1);
    if (magnitude > inf && !(magnitude & (1U << (fraction - 1))))
      *excp_flag_user |= GOC_EXCEPTION_INVALID;
    else if (!(flags & GOC_FP_FLUSH_INPUT_DENORMALS) && magnitude && !(magnitude & inf))
      *excp_flag_user |= GOC_EXCEPTION_INPUT_DENORM;
  }
  *d = goc::round_integral_bits<Op, Half>(a, flags & GOC_FP_FLUSH_INPUT_DENORMALS);
  return GOC_SUCCESS;
}

} // namespace

int goc_s_ceil_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                   uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<IntegralRound::Ceil, false>(flags, instruction_flags, d, a, excp_flag_user);
}

int goc_s_ceil_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                   uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<IntegralRound::Ceil, true>(flags, instruction_flags, d, a, excp_flag_user);
}

int goc_s_floor_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                    uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<IntegralRound::Floor, false>(flags, instruction_flags, d, a, excp_flag_user);
}

int goc_s_floor_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                    uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<IntegralRound::Floor, true>(flags, instruction_flags, d, a, excp_flag_user);
}

int goc_s_trunc_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                    uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<IntegralRound::Trunc, false>(flags, instruction_flags, d, a, excp_flag_user);
}

int goc_s_trunc_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                    uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<IntegralRound::Trunc, true>(flags, instruction_flags, d, a, excp_flag_user);
}

int goc_s_rndne_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                    uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<IntegralRound::Nearest, false>(flags, instruction_flags, d, a, excp_flag_user);
}

int goc_s_rndne_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                    uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<IntegralRound::Nearest, true>(flags, instruction_flags, d, a, excp_flag_user);
}

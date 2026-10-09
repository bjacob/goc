// SPDX-License-Identifier: MIT

// Integer rounding extends rocjitsu util::rndne_scalar to directed rounding.
// GFX1201 verifies all FP16 patterns and raw FP32 boundary/random values.

#include "goc/goc.h"
#include "internal.h"

#include <stdint.h>

namespace {

enum class Round { Ceil, Floor, Trunc, Nearest };

template <Round Op, bool Half> int run(uint64_t flags, uint32_t mode, uint32_t *d, uint32_t a) {
  if (int error = goc::validate(flags, mode, true,
                                GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS))
    return error;
  constexpr unsigned fraction = Half ? 10 : 23, bias = Half ? 15 : 127;
  constexpr uint32_t sign_bit = Half ? 0x8000 : 0x80000000, inf = Half ? 0x7c00 : 0x7f800000;
  if constexpr (Half)
    a &= 65535;
  if ((flags & GOC_FP_FLUSH_INPUT_DENORMALS) && !(a & inf))
    a &= sign_bit;
  uint32_t magnitude = a & (sign_bit - 1), sign = a & sign_bit, result;
  unsigned exponent = magnitude >> fraction;
  if (magnitude >= inf) {
    result = a | (magnitude > inf ? (1u << (fraction - 1)) : 0);
  } else if (exponent >= bias + fraction) {
    result = a;
  } else if (exponent < bias) {
    bool up = false;
    if constexpr (Op == Round::Ceil)
      up = !sign && magnitude;
    if constexpr (Op == Round::Floor)
      up = sign && magnitude;
    if constexpr (Op == Round::Nearest)
      up = magnitude > ((bias - 1) << fraction);
    result = sign | (up ? bias << fraction : 0);
  } else {
    uint32_t unit = 1u << (bias + fraction - exponent), tail = magnitude & (unit - 1),
             rounded = magnitude & ~(unit - 1);
    bool up = false;
    if constexpr (Op == Round::Ceil)
      up = !sign && tail;
    if constexpr (Op == Round::Floor)
      up = sign && tail;
    if constexpr (Op == Round::Nearest)
      up = tail > unit / 2 || (tail == unit / 2 && (rounded & unit));
    result = sign | (rounded + (up ? unit : 0));
  }
  *d = result;
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_s_ceil_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *d, uint32_t a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<Round::Ceil, false>(flags, instruction_flags, d, a);
}

int goc_rdna4_s_ceil_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *d, uint32_t a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<Round::Ceil, true>(flags, instruction_flags, d, a);
}

int goc_rdna4_s_floor_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                          uint32_t *d, uint32_t a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<Round::Floor, false>(flags, instruction_flags, d, a);
}

int goc_rdna4_s_floor_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                          uint32_t *d, uint32_t a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<Round::Floor, true>(flags, instruction_flags, d, a);
}

int goc_rdna4_s_trunc_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                          uint32_t *d, uint32_t a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<Round::Trunc, false>(flags, instruction_flags, d, a);
}

int goc_rdna4_s_trunc_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                          uint32_t *d, uint32_t a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<Round::Trunc, true>(flags, instruction_flags, d, a);
}

int goc_rdna4_s_rndne_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                          uint32_t *d, uint32_t a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<Round::Nearest, false>(flags, instruction_flags, d, a);
}

int goc_rdna4_s_rndne_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                          uint32_t *d, uint32_t a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<Round::Nearest, true>(flags, instruction_flags, d, a);
}

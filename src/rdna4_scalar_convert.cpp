// SPDX-License-Identifier: MIT

// Conversion models follow rocjitsu generated/shared/execute_shared.h and
// util/data_types.h, with GFX1201-verified denormal and NaN handling.

#include "float_formats.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_bits.h"
#include "rdna4_conversion32.h"
#include "rdna4_half_conversion.h"

#include <stdint.h>

namespace {

enum class Convert {
  SignedToFloat,
  UnsignedToFloat,
  FloatToSigned,
  FloatToUnsigned,
  FloatToHalf,
  HalfToFloat,
  HighHalfToFloat,
  PackedHalfRtz
};

template <Convert Op>
int run(uint64_t flags, uint32_t mode, uint32_t *d, uint32_t a, uint32_t b = 0) {
  if (int error = goc::validate(flags, mode, false,
                                GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS))
    return error;
  constexpr bool from_integer = Op == Convert::SignedToFloat || Op == Convert::UnsignedToFloat;
  if constexpr (from_integer) {
    if constexpr (Op == Convert::SignedToFloat)
      *d = goc::as_bits(float(goc::extend_integer<32, true>(a)));
    else
      *d = goc::as_bits(float(a));
  } else {
    constexpr bool half = Op == Convert::HalfToFloat || Op == Convert::HighHalfToFloat;
    if constexpr (Op == Convert::HighHalfToFloat)
      a >>= 16;
    if constexpr (half)
      a &= 65535;
    constexpr uint32_t sign = half ? 0x8000 : 0x80000000, inf = half ? 0x7c00 : 0x7f800000;
    if (flags & GOC_FP_FLUSH_INPUT_DENORMALS) {
      if (!(a & inf))
        a &= sign;
      if (!(b & 0x7f800000))
        b &= 0x80000000;
    }
    if ((a & (sign - 1)) > inf)
      a |= half ? 0x200 : 0x400000;
    float x = half ? goc::f16_to_float(uint16_t(a)) : goc::as_float(a);
    if constexpr (Op == Convert::FloatToSigned) {
      *d = goc::truncate_integer<true>(x);
    } else if constexpr (Op == Convert::FloatToUnsigned) {
      *d = goc::truncate_integer<false>(x);
    } else if constexpr (half) {
      *d = goc::as_bits(x);
    } else {
      uint32_t result;
      if constexpr (Op == Convert::FloatToHalf)
        result = goc::float_to_f16(x, flags & GOC_FP16_OVFL);
      else
        result = goc::half_rtz(a) | (uint32_t(goc::half_rtz(b)) << 16);
      if (flags & GOC_FP_FLUSH_OUTPUT_DENORMALS) {
        bool tiny = !(result & 0x7c00);
        if constexpr (Op == Convert::FloatToHalf)
          tiny |= (result & 0x7fff) == 0x400 && (a & 0x7fffffff) < 0x387ff000;
        if (tiny)
          result &= 0xffff8000u;
        if constexpr (Op == Convert::PackedHalfRtz)
          if (!(result & 0x7c000000))
            result &= 0x8000ffff;
      }
      *d = result;
    }
  }
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_s_cvt_f32_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, uint32_t a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_EXCEPTIONS;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<Convert::SignedToFloat>(flags, instruction_flags, d, a);
}

int goc_rdna4_s_cvt_f32_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, uint32_t a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_EXCEPTIONS;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<Convert::UnsignedToFloat>(flags, instruction_flags, d, a);
}

int goc_rdna4_s_cvt_i32_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, uint32_t a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_EXCEPTIONS;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<Convert::FloatToSigned>(flags, instruction_flags, d, a);
}

int goc_rdna4_s_cvt_u32_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, uint32_t a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_EXCEPTIONS;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<Convert::FloatToUnsigned>(flags, instruction_flags, d, a);
}

int goc_rdna4_s_cvt_f16_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, uint32_t a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_EXCEPTIONS;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<Convert::FloatToHalf>(flags, instruction_flags, d, a);
}

int goc_rdna4_s_cvt_f32_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, uint32_t a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_EXCEPTIONS;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<Convert::HalfToFloat>(flags, instruction_flags, d, a);
}

int goc_rdna4_s_cvt_hi_f32_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, uint32_t a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_EXCEPTIONS;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<Convert::HighHalfToFloat>(flags, instruction_flags, d, a);
}

int goc_rdna4_s_cvt_pk_rtz_f16_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *d, uint32_t a, uint32_t b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_EXCEPTIONS;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<Convert::PackedHalfRtz>(flags, instruction_flags, d, a, b);
}

// SPDX-License-Identifier: MIT

// Fused rounding and tininess detection follow rocjitsu shared/fp_mode.h.
// FP16 uses the existing rocjitsu-derived half_fma_value model.

#include "float_formats.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_half_fma_scalar.h"

#include <cmath>
#include <stdint.h>

namespace {

template <bool Half, bool Accumulate>
int run(uint64_t flags, uint32_t mode, uint32_t *d, uint32_t a, uint32_t b, uint32_t c) {
  if (int error = goc::validate(flags, mode, false,
                                GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS))
    return error;
  if constexpr (Accumulate)
    c = *d;
  constexpr uint32_t sign = Half ? 0x8000 : 0x80000000, inf = Half ? 0x7c00 : 0x7f800000;
  if constexpr (Half) {
    a &= 65535;
    b &= 65535;
    c &= 65535;
  }
  if (flags & GOC_FP_FLUSH_INPUT_DENORMALS) {
    if (!(a & inf))
      a &= sign;
    if (!(b & inf))
      b &= sign;
    if (!(c & inf))
      c &= sign;
  }
  float x = Half ? goc::f16_to_float(uint16_t(a)) : goc::as_float(a);
  float y = Half ? goc::f16_to_float(uint16_t(b)) : goc::as_float(b);
  float z = Half ? goc::f16_to_float(uint16_t(c)) : goc::as_float(c);
  uint32_t result;
  if constexpr (Half)
    result = goc::half_fma_value(uint16_t(a), uint16_t(b), uint16_t(c), 0, flags & GOC_FP16_OVFL);
  else
    result = goc::as_bits(std::fma(x, y, z));
  if (flags & GOC_FP_FLUSH_OUTPUT_DENORMALS) {
    uint32_t magnitude = result & (sign - 1);
    constexpr uint32_t normal = Half ? 0x400 : 0x800000;
    if (magnitude < normal)
      result &= sign;
    if (magnitude == normal) {
      // TwoSum retains tiny addends when deciding which side of the tininess
      // threshold the exact result occupies. The product is exact in double.
      double product = double(x) * double(y), sum = product + double(z), part = sum - product;
      double residual = (product - (sum - part)) + (double(z) - part);
      double absolute = (result & sign) ? -sum : sum, tail = (result & sign) ? -residual : residual;
      constexpr double threshold = Half ? (0x1p-14 - 0x1p-26) : (0x1p-126 - 0x1p-151);
      if (absolute < threshold || (absolute == threshold && tail < 0))
        result &= sign;
    }
  }
  *d = result;
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_s_fmac_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         uint32_t b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<false, true>(flags, instruction_flags, d, a, b, 0);
}

int goc_rdna4_s_fmac_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         uint32_t b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<true, true>(flags, instruction_flags, d, a, b, 0);
}

int goc_rdna4_s_fmaak_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                          uint32_t b, uint32_t literal, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<false, false>(flags, instruction_flags, d, a, b, literal);
}

int goc_rdna4_s_fmamk_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                          uint32_t literal, uint32_t c, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<false, false>(flags, instruction_flags, d, a, literal, c);
}

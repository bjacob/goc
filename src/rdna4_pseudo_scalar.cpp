// SPDX-License-Identifier: MIT

// Stage ordering follows rocjitsu shared/pseudo_scalar.h and transcendental.h.
// GFX1201 captures correct half OMOD underflow: newly created -0 keeps its sign.

#include "float_formats.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_half_fma_scalar.h"
#include "rdna4_unary.h"

#include <algorithm>
#include <cmath>
#include <stdint.h>

namespace {

template <goc::Unary Op, bool Half>
int run(uint64_t flags, uint32_t mode, uint32_t *d, uint32_t raw) {
  const uint32_t known = GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP;
  if (int error = goc::validate(flags, mode & ~known, false,
                                GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS))
    return error;
  constexpr uint32_t sign = Half ? 0x8000 : 0x80000000, mag = sign - 1,
                     inf = Half ? 0x7c00 : 0x7f800000;
  if constexpr (Half)
    raw &= 65535;
  if (mode & GOC_ALU_ABS_A)
    raw &= mag;
  if (mode & GOC_ALU_NEG_A)
    raw ^= sign;
  const bool finite = (raw & mag) < inf;
  if ((!Half || (flags & GOC_FP_FLUSH_INPUT_DENORMALS)) && !(raw & inf))
    raw &= sign;
  double x = Half ? goc::f16_to_float(uint16_t(raw)) : goc::as_float(raw), value;
  if constexpr (Op == goc::Unary::Exp)
    value = std::exp2(std::isfinite(x) ? std::clamp(x, -256., 256.) : x);
  if constexpr (Op == goc::Unary::Log)
    value = std::log2(x);
  if constexpr (Op == goc::Unary::Rcp)
    value = 1 / x;
  if constexpr (Op == goc::Unary::Rsq)
    value = 1 / std::sqrt(x);
  if constexpr (Op == goc::Unary::Sqrt)
    value = std::sqrt(x);
  const unsigned omod = (mode >> 6) & 3;
  if constexpr (Half) {
    bool ovfl = flags & GOC_FP16_OVFL;
    uint16_t result = std::isfinite(value) ? goc::half_fma_narrow(value, ovfl)
                                           : goc::float_to_f16(float(value), false);
    if (ovfl && finite && std::isinf(value))
      result = (std::signbit(value) ? 0x8000 : 0) | 0x7bff;
    if ((flags & GOC_FP_FLUSH_OUTPUT_DENORMALS) && !(result & 0x7c00))
      result &= 0x8000;
    if (omod) {
      if ((result & 0x7fff) < 0x400)
        result = 0;
      float scaled = goc::f16_to_float(result) * (omod == 1 ? 2 : omod == 2 ? 4 : 0.5f);
      result = goc::float_to_f16(scaled, ovfl);
      if (!(result & 0x7c00))
        result &= 0x8000;
    }
    if (mode & GOC_ALU_CLAMP)
      result = goc::half_fma_clamp(result);
    *d = result;
  } else {
    float rounded = float(value);
    uint32_t result = goc::as_bits(rounded);
    if (!(result & 0x7f800000))
      result &= 0x80000000;
    rounded = goc::as_float(result);
    if (omod) {
      rounded *= omod == 1 ? 2 : omod == 2 ? 4 : 0.5f;
      if (rounded == 0)
        rounded = 0;
    }
    if (mode & GOC_ALU_CLAMP)
      rounded = !(rounded > 0) ? 0 : rounded > 1 ? 1 : rounded;
    result = goc::as_bits(rounded);
    if (!(result & 0x7f800000))
      result &= 0x80000000;
    *d = result;
  }
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_s_exp_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                          uint32_t *d, uint32_t a) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<goc::Unary::Exp, false>(flags, instruction_flags, d, a);
}

int goc_rdna4_v_s_exp_f16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                          uint32_t *d, uint32_t a) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<goc::Unary::Exp, true>(flags, instruction_flags, d, a);
}

int goc_rdna4_v_s_log_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                          uint32_t *d, uint32_t a) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<goc::Unary::Log, false>(flags, instruction_flags, d, a);
}

int goc_rdna4_v_s_log_f16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                          uint32_t *d, uint32_t a) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<goc::Unary::Log, true>(flags, instruction_flags, d, a);
}

int goc_rdna4_v_s_rcp_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                          uint32_t *d, uint32_t a) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<goc::Unary::Rcp, false>(flags, instruction_flags, d, a);
}

int goc_rdna4_v_s_rcp_f16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                          uint32_t *d, uint32_t a) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<goc::Unary::Rcp, true>(flags, instruction_flags, d, a);
}

int goc_rdna4_v_s_rsq_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                          uint32_t *d, uint32_t a) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<goc::Unary::Rsq, false>(flags, instruction_flags, d, a);
}

int goc_rdna4_v_s_rsq_f16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                          uint32_t *d, uint32_t a) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<goc::Unary::Rsq, true>(flags, instruction_flags, d, a);
}

int goc_rdna4_v_s_sqrt_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, uint32_t a) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<goc::Unary::Sqrt, false>(flags, instruction_flags, d, a);
}

int goc_rdna4_v_s_sqrt_f16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, uint32_t a) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<goc::Unary::Sqrt, true>(flags, instruction_flags, d, a);
}

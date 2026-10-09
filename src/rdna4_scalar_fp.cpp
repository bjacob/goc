// SPDX-License-Identifier: MIT

// Scalar operations follow rocjitsu generated/shared/execute_shared.h. GFX1201
// captures establish the output-flush exceptions and multiplication threshold.

#include "float_formats.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_binary.h"
#include "rdna4_minmax.h"

#include <cmath>
#include <stdint.h>

namespace {

template <goc::Binary Op, bool Half>
int run(uint64_t flags, uint32_t mode, uint32_t *d, uint32_t a, uint32_t b) {
  if (int error = goc::validate(flags, mode, false,
                                GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS))
    return error;
  constexpr uint32_t inf = Half ? 0x7c00 : 0x7f800000, sign = Half ? 0x8000 : 0x80000000;
  if constexpr (Half) {
    a &= 65535;
    b &= 65535;
  }
  if (flags & GOC_FP_FLUSH_INPUT_DENORMALS) {
    if (!(a & inf))
      a &= sign;
    if (!(b & inf))
      b &= sign;
  }
  float x = Half ? goc::f16_to_float(uint16_t(a)) : goc::as_float(a);
  float y = Half ? goc::f16_to_float(uint16_t(b)) : goc::as_float(b), value;
  if constexpr (Op == goc::Binary::Add)
    value = x + y;
  if constexpr (Op == goc::Binary::Sub)
    value = x - y;
  if constexpr (Op == goc::Binary::Mul)
    value = x * y;
  if constexpr (Op == goc::Binary::MinNum)
    value = goc::minmax<false, false>(x, y);
  if constexpr (Op == goc::Binary::MaxNum)
    value = goc::minmax<true, false>(x, y);
  if constexpr (Op == goc::Binary::Minimum)
    value = goc::minmax<false, true>(x, y);
  if constexpr (Op == goc::Binary::Maximum)
    value = goc::minmax<true, true>(x, y);
  uint32_t result = Half ? goc::float_to_f16(value, flags & GOC_FP16_OVFL) : goc::as_bits(value);
  if constexpr (Op == goc::Binary::Add || Op == goc::Binary::Sub || Op == goc::Binary::Mul) {
    if (flags & GOC_FP_FLUSH_OUTPUT_DENORMALS) {
      if (!(result & inf))
        result &= sign;
      if constexpr (Op == goc::Binary::Mul) {
        // GPU tininess detection precedes destination subnormal rounding. The
        // threshold is a quarter subnormal ULP below the smallest normal; ties
        // round to the normal value. Double holds the exact FP32 product.
        constexpr double threshold = Half ? (0x1p-14 - 0x1p-26) : (0x1p-126 - 0x1p-151);
        if (std::abs(double(x) * double(y)) < threshold)
          result &= sign;
      }
    }
  }
  *d = result;
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_s_add_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                        uint32_t a, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<goc::Binary::Add, false>(flags, instruction_flags, d, a, b);
}

int goc_rdna4_s_add_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                        uint32_t a, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<goc::Binary::Add, true>(flags, instruction_flags, d, a, b);
}

int goc_rdna4_s_sub_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                        uint32_t a, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<goc::Binary::Sub, false>(flags, instruction_flags, d, a, b);
}

int goc_rdna4_s_sub_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                        uint32_t a, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<goc::Binary::Sub, true>(flags, instruction_flags, d, a, b);
}

int goc_rdna4_s_mul_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                        uint32_t a, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<goc::Binary::Mul, false>(flags, instruction_flags, d, a, b);
}

int goc_rdna4_s_mul_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                        uint32_t a, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<goc::Binary::Mul, true>(flags, instruction_flags, d, a, b);
}

int goc_rdna4_s_min_num_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, uint32_t a, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<goc::Binary::MinNum, false>(flags, instruction_flags, d, a, b);
}

int goc_rdna4_s_min_num_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, uint32_t a, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<goc::Binary::MinNum, true>(flags, instruction_flags, d, a, b);
}

int goc_rdna4_s_max_num_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, uint32_t a, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<goc::Binary::MaxNum, false>(flags, instruction_flags, d, a, b);
}

int goc_rdna4_s_max_num_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, uint32_t a, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<goc::Binary::MaxNum, true>(flags, instruction_flags, d, a, b);
}

int goc_rdna4_s_minimum_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, uint32_t a, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<goc::Binary::Minimum, false>(flags, instruction_flags, d, a, b);
}

int goc_rdna4_s_minimum_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, uint32_t a, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<goc::Binary::Minimum, true>(flags, instruction_flags, d, a, b);
}

int goc_rdna4_s_maximum_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, uint32_t a, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<goc::Binary::Maximum, false>(flags, instruction_flags, d, a, b);
}

int goc_rdna4_s_maximum_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, uint32_t a, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return run<goc::Binary::Maximum, true>(flags, instruction_flags, d, a, b);
}

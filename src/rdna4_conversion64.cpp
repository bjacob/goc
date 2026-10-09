// SPDX-License-Identifier: MIT

#include "rdna4_conversion64.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_alu.h"
#include "rdna4_fp64.h"

#include <cmath>
#include <stdint.h>

namespace {

template <goc::Conversion64 Op>
int convert(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
            const uint32_t *const *a) {
  constexpr bool from_integer =
      Op == goc::Conversion64::SignedToDouble || Op == goc::Conversion64::UnsignedToDouble;
  constexpr bool to_double = from_integer || Op == goc::Conversion64::FloatToDouble;
  const uint32_t known =
      GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP | (from_integer ? 0 : GOC_ALU_NEG_A | GOC_ALU_ABS_A);
  if (int error = goc::validate(flags, mode & ~known))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::conversion64_x86_64_v4<Op>(exec_mask, mode, d, a);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  // The AVX2 signed truncating candidate was roughly tied with baseline.
  if constexpr (Op != goc::Conversion64::DoubleToSigned) {
    if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
      goc::conversion64_x86_64_v3<Op>(exec_mask, mode, d, a);
      return GOC_SUCCESS;
    }
  }
#endif
  uint32_t result[to_double ? 2 : 1][32];
  for (int lane = 0; lane < 32; ++lane) {
    if constexpr (to_double) {
      double value;
      if constexpr (Op == goc::Conversion64::SignedToDouble) {
        uint32_t raw = a[0][lane];
        value = double(int64_t(raw) - (raw >> 31 ? INT64_C(4294967296) : 0));
      } else if constexpr (Op == goc::Conversion64::UnsignedToDouble)
        value = double(a[0][lane]);
      else
        value = double(goc::alu_input(a[0][lane], mode));
      uint64_t bits = goc::double_bits(goc::fp64_output(value, mode));
      result[0][lane] = uint32_t(bits);
      result[1][lane] = uint32_t(bits >> 32);
    } else {
      double value = goc::fp64_input(a, lane, mode);
      if constexpr (Op == goc::Conversion64::DoubleToFloat) {
        // Round to FP32 before output scaling, as in rocjitsu's conversion
        // semantics. A later scale-down must not undo conversion overflow.
        // Active OMOD flushes tininess before narrowing, even when rounding
        // would otherwise produce minimum normal.
        if ((mode & GOC_ALU_OMOD_HALF) && std::abs(value) < 0x1p-126)
          value = 0;
        result[0][lane] = goc::as_bits(goc::alu_output_f32(float(value), mode));
      } else if constexpr (Op == goc::Conversion64::DoubleToUnsigned) {
        // Match rocjitsu's sema_lower.py: truncate, saturate, map NaN to zero.
        // Classify before casting, to avoid undefined out-of-range casts.
        result[0][lane] = !(value > 0) ? 0 : value >= 4294967296.0 ? UINT32_MAX : uint32_t(value);
      } else {
        result[0][lane] = std::isnan(value)        ? 0
                          : value >= 2147483648.0  ? uint32_t(INT32_MAX)
                          : value <= -2147483648.0 ? uint32_t(INT32_MIN)
                                                   : uint32_t(int32_t(value));
      }
      // Integer outputs ignore CLAMP and OMOD numerically. GPU exception
      // reporting is not part of the host FP environment contract.
    }
  }
  for (int reg = 0; reg < (to_double ? 2 : 1); ++reg)
    for (int lane = 0; lane < 32; ++lane)
      if ((exec_mask >> lane) & 1)
        d[reg][lane] = result[reg][lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_cvt_f64_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return convert<goc::Conversion64::SignedToDouble>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_cvt_f64_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return convert<goc::Conversion64::UnsignedToDouble>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_cvt_i32_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return convert<goc::Conversion64::DoubleToSigned>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_cvt_u32_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return convert<goc::Conversion64::DoubleToUnsigned>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_cvt_f64_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return convert<goc::Conversion64::FloatToDouble>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_cvt_f32_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return convert<goc::Conversion64::DoubleToFloat>(flags, exec_mask, instruction_flags, d, a);
}

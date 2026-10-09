// SPDX-License-Identifier: MIT

#include "conversion32.h"
#include "alu.h"
#include "bits.h"
#include "dpp.h"
#include "goc/goc.h"
#include "internal.h"

#include <cmath>
#include <stdint.h>

namespace {

template <goc::Conversion32 Op>
int convert(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a) {
  if (mode >> 32)
    return goc::execute_dpp(flags, exec_mask, mode, a,
                            [&](uint32_t exec_mask, const uint32_t *const *source) {
                              return convert<Op>(flags, exec_mask, uint32_t(mode), d, source);
                            });
  constexpr bool to_float =
      Op == goc::Conversion32::SignedToFloat || Op == goc::Conversion32::UnsignedToFloat;
  const uint32_t known =
      GOC_ALU_CLAMP | (to_float ? 0 : GOC_ALU_ABS_A | GOC_ALU_NEG_A) |
      (Op == goc::Conversion32::Nearest || Op == goc::Conversion32::Floor ? 0 : GOC_ALU_OMOD_HALF);
  if (int error = goc::validate(flags, mode & ~known))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::conversion32_x86_64_v4<Op>(exec_mask, mode, d[0], a[0]);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  // AVX2 truncating conversions gained only 7-10% over baseline. Keep the
  // smaller baseline path there; AVX-512 still accelerates every opcode.
  if constexpr (Op != goc::Conversion32::FloatToSigned &&
                Op != goc::Conversion32::FloatToUnsigned) {
    if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
      goc::conversion32_x86_64_v3<Op>(exec_mask, mode, d[0], a[0]);
      return GOC_SUCCESS;
    }
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    uint32_t raw = a[0][lane];
    if constexpr (to_float) {
      float x;
      if constexpr (Op == goc::Conversion32::SignedToFloat)
        x = float(goc::extend_integer<32, true>(raw));
      else
        x = float(raw);
      result[lane] = goc::as_bits(goc::alu_output(x, mode));
    } else {
      // Saturation and nearest ties toward +infinity follow rocjitsu's
      // codegen/execute/sema_lower.py conversion models. Classify before
      // casting: out-of-range floating-to-integer casts are undefined in C++.
      float x = goc::alu_input(raw, mode);
      uint32_t nan_result = 0;
      if constexpr (Op == goc::Conversion32::Nearest || Op == goc::Conversion32::Floor)
        nan_result = (goc::as_bits(x) >> 31) ? uint32_t(INT32_MIN) : uint32_t(INT32_MAX);
      if constexpr (Op == goc::Conversion32::Nearest) {
        float rounded = std::floor(x);
        if (x - rounded >= 0.5f)
          rounded += 1;
        x = rounded;
      }
      if constexpr (Op == goc::Conversion32::Floor)
        x = std::floor(x);
      result[lane] = goc::truncate_integer<Op != goc::Conversion32::FloatToUnsigned>(x, nan_result);
      // Integer CLAMP and the truncating opcodes' OMOD do not scale or
      // clamp numeric results. GPU exception reporting is not modeled.
    }
  }
  for (int lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_v_cvt_f32_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a) {
  return convert<goc::Conversion32::SignedToFloat>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_cvt_f32_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a) {
  return convert<goc::Conversion32::UnsignedToFloat>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_cvt_i32_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return convert<goc::Conversion32::FloatToSigned>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_cvt_u32_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return convert<goc::Conversion32::FloatToUnsigned>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_cvt_nearest_i32_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a) {
  return convert<goc::Conversion32::Nearest>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_cvt_floor_i32_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a) {
  return convert<goc::Conversion32::Floor>(flags, exec_mask, instruction_flags, d, a);
}

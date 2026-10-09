// SPDX-License-Identifier: MIT

#include "rdna4_conversion16.h"
#include "float_formats.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_alu.h"

#include <cmath>
#include <stdint.h>

namespace {

uint16_t half_output(float value, bool saturate, uint32_t mode) {
  // Round to half before scaling, with the captured RDNA4 tininess rule.
  uint16_t result = goc::float_to_f16(value, saturate);
  unsigned omod = (mode >> 6) & 3;
  if (omod) {
    // 0x0400 is the smallest normal half. At that boundary the tininess
    // threshold is 2^-14 - 2^-26, encoded as FP32 0x387ff000.
    uint16_t magnitude = result & 0x7fff;
    if (magnitude < 0x400 ||
        (magnitude == 0x400 && (goc::as_bits(value) & 0x7fffffff) < 0x387ff000))
      result = 0;
    else if (omod == 3 && magnitude < 0x800)
      result &= 0x8000;
    else {
      const float scales[] = {1, 2, 4, 0.5f};
      result = goc::float_to_f16(goc::f16_to_float(result) * scales[omod], saturate);
    }
  }
  if (mode & GOC_ALU_CLAMP) {
    if ((result & 0x8000) || (result & 0x7fff) > 0x7c00)
      result = 0;
    else if (result > 0x3c00)
      result = 0x3c00;
  }
  return result;
}

template <goc::Conversion16 Op>
int convert(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
            const uint32_t *const *a) {
  constexpr bool from_integer =
      Op == goc::Conversion16::SignedToHalf || Op == goc::Conversion16::UnsignedToHalf;
  constexpr bool to_half = from_integer || Op == goc::Conversion16::FloatToHalf;
  const uint32_t known = GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP |
                         (from_integer ? 0 : GOC_ALU_ABS_A | GOC_ALU_NEG_A) |
                         (Op == goc::Conversion16::FloatToHalf ? 0 : GOC_ALU_HIGH_A) |
                         (Op == goc::Conversion16::HalfToFloat ? 0 : GOC_ALU_HIGH_D);
  if (int error = goc::validate(flags, mode & ~known))
    return error;
  if (!uint32_t(mask))
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::conversion16_x86_64_v4<Op>(flags & GOC_FP16_OVFL, uint32_t(mask), mode, d[0], a[0]);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::conversion16_x86_64_v3<Op>(flags & GOC_FP16_OVFL, uint32_t(mask), mode, d[0], a[0]);
    return GOC_SUCCESS;
  }
#endif
  unsigned sa = mode & GOC_ALU_HIGH_A ? 16 : 0, sd = mode & GOC_ALU_HIGH_D ? 16 : 0;
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    uint32_t raw = a[0][lane];
    float value;
    if constexpr (from_integer) {
      uint16_t selected = uint16_t(raw >> sa);
      value = Op == goc::Conversion16::SignedToHalf
                  ? float(int(selected) - ((selected >> 15) ? 65536 : 0))
                  : float(selected);
    } else {
      if constexpr (Op != goc::Conversion16::FloatToHalf)
        raw = goc::as_bits(goc::f16_to_float(uint16_t(raw >> sa)));
      if ((raw & 0x7fffffff) > 0x7f800000)
        raw |= 0x400000; // RDNA4 conversions quiet signaling NaNs.
      value = goc::alu_input(raw, mode);
    }
    if constexpr (to_half)
      result[lane] = half_output(value, flags & GOC_FP16_OVFL, mode);
    else if constexpr (Op == goc::Conversion16::HalfToFloat) {
      value = goc::alu_output(value, mode);
      if ((mode & GOC_ALU_OMOD_HALF) && value == 0)
        value = 0;
      result[lane] = goc::as_bits(value);
    } else if constexpr (Op == goc::Conversion16::HalfToUnsigned)
      result[lane] = !(value > 0) ? 0 : value >= 65536 ? 65535 : uint16_t(value);
    else
      result[lane] = std::isnan(value) ? 0
                     : value >= 32768  ? 32767
                     : value <= -32768 ? 32768
                                       : uint16_t(int16_t(value));
    if constexpr (Op != goc::Conversion16::HalfToFloat)
      result[lane] = (d[0][lane] & ~(uint32_t(65535) << sd)) | (result[lane] << sd);
  }
  for (int lane = 0; lane < 32; ++lane)
    if ((mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_cvt_f16_i16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a) {
  return convert<goc::Conversion16::SignedToHalf>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_cvt_f16_u16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a) {
  return convert<goc::Conversion16::UnsignedToHalf>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_cvt_i16_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a) {
  return convert<goc::Conversion16::HalfToSigned>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_cvt_u16_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a) {
  return convert<goc::Conversion16::HalfToUnsigned>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_cvt_f16_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a) {
  return convert<goc::Conversion16::FloatToHalf>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_cvt_f32_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a) {
  return convert<goc::Conversion16::HalfToFloat>(flags, exec_mask, instruction_flags, d, a);
}

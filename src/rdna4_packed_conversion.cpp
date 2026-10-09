// SPDX-License-Identifier: MIT

#include "rdna4_packed_conversion.h"
#include "goc/goc.h"
#include "internal.h"

#include <cmath>
#include <stdint.h>

namespace {

// Round toward zero to FP16, saturating finite overflow and quieting NaNs.
uint16_t half_rtz(uint32_t raw) {
  // Adapted from rocjitsu's util::f32_to_f16_rtz; RDNA4 quiets source NaNs.
  uint32_t sign = (raw >> 16) & 0x8000;
  unsigned exponent = (raw >> 23) & 255, fraction = raw & 0x7fffff;
  if (exponent == 255)
    return uint16_t(sign | 0x7c00 | (fraction ? (fraction >> 13) | 0x200 : 0));
  int adjusted = int(exponent) - 112;
  if (adjusted <= 0)
    return uint16_t(sign | (adjusted < -10 ? 0 : (fraction | 0x800000) >> (14 - adjusted)));
  if (adjusted >= 31)
    return uint16_t(sign | 0x7bff);
  return uint16_t(sign | (unsigned(adjusted) << 10) | (fraction >> 13));
}

template <goc::PackedConversion Op> uint16_t narrow(uint32_t raw, uint32_t mode) {
  if (mode & GOC_ALU_ABS_A)
    raw &= 0x7fffffff;
  if (mode & GOC_ALU_NEG_A)
    raw ^= 0x80000000;
  if constexpr (Op == goc::PackedConversion::HalfRtz)
    return half_rtz(raw);
  else {
    float value = goc::as_float(raw);
    if constexpr (Op == goc::PackedConversion::Unsigned)
      return !(value > 0) ? 0 : value >= 65535 ? 65535 : uint16_t(value);
    else
      return std::isnan(value) ? 0
             : value >= 32767  ? 32767
             : value <= -32768 ? 32768
                               : uint16_t(int16_t(value));
  }
}

template <goc::PackedConversion Op>
int convert(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b) {
  const uint32_t known = GOC_ALU_ABS_A | GOC_ALU_ABS_B | GOC_ALU_NEG_A | GOC_ALU_NEG_B |
                         GOC_ALU_CLAMP |
                         (Op == goc::PackedConversion::HalfRtz ? GOC_ALU_OMOD_HALF : 0);
  if (int error = goc::validate(flags, mode & ~known))
    return error;
  if (!uint32_t(mask))
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::packed_conversion_x86_64_v4<Op>(uint32_t(mask), mode, d[0], a[0], b[0]);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::packed_conversion_x86_64_v3<Op>(uint32_t(mask), mode, d[0], a[0], b[0]);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (unsigned lane = 0; lane < 32; ++lane)
    result[lane] =
        narrow<Op>(a[0][lane], mode) | (uint32_t(narrow<Op>(b[0][lane], mode >> 1)) << 16);
  for (unsigned lane = 0; lane < 32; ++lane)
    if ((mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_cvt_pk_rtz_f16_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b) {
  return convert<goc::PackedConversion::HalfRtz>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cvt_pk_i16_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b) {
  return convert<goc::PackedConversion::Signed>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cvt_pk_u16_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b) {
  return convert<goc::PackedConversion::Unsigned>(flags, exec_mask, instruction_flags, d, a, b);
}

// SPDX-License-Identifier: MIT

#include "rdna4_normalized.h"
#include "float_formats.h"
#include "goc/goc.h"
#include "internal.h"

#include <cmath>
#include <stdint.h>

namespace {

template <bool Unsigned, bool Half> uint16_t normalized(uint32_t raw, uint32_t mode) {
  if constexpr (Half)
    raw = goc::as_bits(goc::f16_to_float(uint16_t(raw >> (mode & GOC_ALU_HIGH_A ? 16 : 0))));
  if (mode & GOC_ALU_ABS_A)
    raw &= 0x7fffffff;
  if (mode & GOC_ALU_NEG_A)
    raw ^= 0x80000000;
  double value = goc::as_float(raw);
  if (std::isnan(value))
    return 0;
  constexpr double scale = Unsigned ? 65535 : 32767;
  value = value < (Unsigned ? 0 : -1) ? (Unsigned ? 0 : -1) : value > 1 ? 1 : value;
  // rocjitsu's normalized model uses an exact double product before rounding
  // to nearest-even, avoiding FP32 products that round onto a false tie.
  double product = value * scale;
  double lower = std::floor(product), fraction = product - lower;
  int32_t result = int32_t(lower);
  result += fraction > 0.5 || (fraction == 0.5 && (uint32_t(result) & 1));
  return uint16_t(result);
}

template <bool Unsigned, goc::NormalizedForm Form>
int convert(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b) {
  constexpr bool unary = Form == goc::NormalizedForm::Half;
  constexpr bool half = Form != goc::NormalizedForm::PackedFloat;
  const uint32_t known = GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_CLAMP |
                         (half ? GOC_ALU_HIGH_A : 0) |
                         (unary ? GOC_ALU_HIGH_D | GOC_ALU_OMOD_HALF
                                : GOC_ALU_ABS_B | GOC_ALU_NEG_B | (half ? GOC_ALU_HIGH_B : 0));
  if (int error = goc::validate(flags, mode & ~known))
    return error;
  if (!uint32_t(mask))
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::normalized_x86_64_v4<Unsigned, Form>(uint32_t(mask), mode, d[0], a[0],
                                              unary ? nullptr : b[0]);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::normalized_x86_64_v3<Unsigned, Form>(uint32_t(mask), mode, d[0], a[0],
                                              unary ? nullptr : b[0]);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[32];
  unsigned shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  for (unsigned lane = 0; lane < 32; ++lane) {
    uint32_t low = normalized<Unsigned, half>(a[0][lane], mode);
    if constexpr (unary)
      result[lane] = (d[0][lane] & ~(65535u << shift)) | (low << shift);
    else
      result[lane] = low | (uint32_t(normalized<Unsigned, half>(b[0][lane], mode >> 1)) << 16);
  }
  for (unsigned lane = 0; lane < 32; ++lane)
    if ((mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_cvt_pk_norm_i16_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b) {
  return convert<false, goc::NormalizedForm::PackedFloat>(flags, exec_mask, instruction_flags, d, a,
                                                          b);
}

int goc_rdna4_v_cvt_pk_norm_u16_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b) {
  return convert<true, goc::NormalizedForm::PackedFloat>(flags, exec_mask, instruction_flags, d, a,
                                                         b);
}

int goc_rdna4_v_cvt_pk_norm_i16_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b) {
  return convert<false, goc::NormalizedForm::PackedHalf>(flags, exec_mask, instruction_flags, d, a,
                                                         b);
}

int goc_rdna4_v_cvt_pk_norm_u16_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b) {
  return convert<true, goc::NormalizedForm::PackedHalf>(flags, exec_mask, instruction_flags, d, a,
                                                        b);
}

int goc_rdna4_v_cvt_norm_i16_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a) {
  return convert<false, goc::NormalizedForm::Half>(flags, exec_mask, instruction_flags, d, a,
                                                   nullptr);
}

int goc_rdna4_v_cvt_norm_u16_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a) {
  return convert<true, goc::NormalizedForm::Half>(flags, exec_mask, instruction_flags, d, a,
                                                  nullptr);
}

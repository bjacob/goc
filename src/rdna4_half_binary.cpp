// SPDX-License-Identifier: MIT

#include "rdna4_half_binary.h"
#include "float_formats.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_alu.h"
#include "rdna4_binary.h"
#include "rdna4_minmax.h"
#include "rdna4_packed_alu.h"

#include <stdint.h>

namespace {

template <goc::Binary Op, bool Packed = false>
int binary(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
           const uint32_t *const *a, const uint32_t *const *b) {
  const uint32_t known = Packed ? GOC_PK_NEG_LO_A | GOC_PK_NEG_LO_B | GOC_PK_NEG_HI_A |
                                      GOC_PK_NEG_HI_B | GOC_PK_CLAMP | GOC_PK_LO_A_HIGH |
                                      GOC_PK_LO_B_HIGH | GOC_PK_HI_A_LOW | GOC_PK_HI_B_LOW
                                : GOC_ALU_NEG_A | GOC_ALU_NEG_B | GOC_ALU_ABS_A | GOC_ALU_ABS_B |
                                      GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP | GOC_ALU_HIGH_A |
                                      GOC_ALU_HIGH_B | GOC_ALU_HIGH_D;
  if (int error = goc::validate(flags, mode & ~known))
    return error;
  if (uint32_t(mask) == 0)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::half_binary_x86_64_v3<Op, Packed>(bool(flags & GOC_FP16_OVFL), uint32_t(mask), mode, d[0],
                                           a[0], b[0]);
    return GOC_SUCCESS;
  }
#endif
  uint32_t modes[] = {Packed ? goc::packed_half_mode(mode, false) : mode,
                      goc::packed_half_mode(mode, true)};
  int d_shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    result[lane] = 0;
    for (int half = 0; half < (Packed ? 2 : 1); ++half) {
      uint32_t m = modes[half];
      int a_shift = m & GOC_ALU_HIGH_A ? 16 : 0;
      int b_shift = m & GOC_ALU_HIGH_B ? 16 : 0;
      float x = goc::alu_input(goc::as_bits(goc::f16_to_float(uint16_t(a[0][lane] >> a_shift))), m);
      float y =
          goc::alu_input(goc::as_bits(goc::f16_to_float(uint16_t(b[0][lane] >> b_shift))), m >> 1);
      float value;
      if constexpr (Op == goc::Binary::Add)
        value = x + y;
      if constexpr (Op == goc::Binary::Sub)
        value = x - y;
      if constexpr (Op == goc::Binary::Subrev)
        value = y - x;
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
      result[lane] |= uint32_t(goc::float_to_f16(goc::alu_output(value, m), flags & GOC_FP16_OVFL))
                      << (16 * half);
    }
  }
  for (int lane = 0; lane < 32; ++lane)
    if ((mask >> lane) & 1) {
      if constexpr (Packed)
        d[0][lane] = result[lane];
      else
        d[0][lane] = (d[0][lane] & ~(UINT32_C(0xffff) << d_shift)) | (result[lane] << d_shift);
    }
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_add_f16(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b) {
  return binary<goc::Binary::Add>(flags, mask, mode, d, a, b);
}

int goc_rdna4_v_sub_f16(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b) {
  return binary<goc::Binary::Sub>(flags, mask, mode, d, a, b);
}

int goc_rdna4_v_subrev_f16(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b) {
  return binary<goc::Binary::Subrev>(flags, mask, mode, d, a, b);
}

int goc_rdna4_v_mul_f16(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b) {
  return binary<goc::Binary::Mul>(flags, mask, mode, d, a, b);
}

int goc_rdna4_v_min_num_f16(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b) {
  return binary<goc::Binary::MinNum>(flags, mask, mode, d, a, b);
}

int goc_rdna4_v_max_num_f16(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b) {
  return binary<goc::Binary::MaxNum>(flags, mask, mode, d, a, b);
}

int goc_rdna4_v_minimum_f16(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b) {
  return binary<goc::Binary::Minimum>(flags, mask, mode, d, a, b);
}

int goc_rdna4_v_maximum_f16(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b) {
  return binary<goc::Binary::Maximum>(flags, mask, mode, d, a, b);
}

int goc_rdna4_v_pk_add_f16(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b) {
  return binary<goc::Binary::Add, true>(flags, mask, mode, d, a, b);
}

int goc_rdna4_v_pk_mul_f16(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b) {
  return binary<goc::Binary::Mul, true>(flags, mask, mode, d, a, b);
}

int goc_rdna4_v_pk_min_num_f16(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                               const uint32_t *const *a, const uint32_t *const *b) {
  return binary<goc::Binary::MinNum, true>(flags, mask, mode, d, a, b);
}

int goc_rdna4_v_pk_max_num_f16(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                               const uint32_t *const *a, const uint32_t *const *b) {
  return binary<goc::Binary::MaxNum, true>(flags, mask, mode, d, a, b);
}

int goc_rdna4_v_pk_minimum_f16(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                               const uint32_t *const *a, const uint32_t *const *b) {
  return binary<goc::Binary::Minimum, true>(flags, mask, mode, d, a, b);
}

int goc_rdna4_v_pk_maximum_f16(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                               const uint32_t *const *a, const uint32_t *const *b) {
  return binary<goc::Binary::Maximum, true>(flags, mask, mode, d, a, b);
}

// SPDX-License-Identifier: MIT

#include "rdna4_integer16_ternary.h"
#include "goc/goc.h"
#include "internal.h"

#include <algorithm>
#include <stdint.h>

namespace {

template <bool Signed> int64_t input(uint32_t word, int half) {
  int64_t value = uint16_t(word >> (16 * half));
  if constexpr (Signed)
    value -= (value & 0x8000) ? 65536 : 0;
  return value;
}

template <bool Signed, goc::Integer16Ternary Op = goc::Integer16Ternary::Mad, bool Packed = true>
int ternary(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  const uint32_t known =
      Packed ? GOC_PK_LO_A_HIGH | GOC_PK_LO_B_HIGH | GOC_PK_LO_C_HIGH | GOC_PK_HI_A_LOW |
                   GOC_PK_HI_B_LOW | GOC_PK_HI_C_LOW | GOC_PK_CLAMP
             : GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_C | GOC_ALU_HIGH_D |
                   (Op == goc::Integer16Ternary::Mad ? GOC_ALU_CLAMP : 0);
  if (int error = goc::validate(flags, mode & ~known))
    return error;
  if (uint32_t(mask) == 0)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::integer16_ternary_x86_64_v3<Signed, Op, Packed>(uint32_t(mask), mode, d[0], a[0], b[0],
                                                         c[0]);
    return GOC_SUCCESS;
  }
#endif
  uint32_t arithmetic_mode = mode;
  if constexpr (!Packed)
    arithmetic_mode = (mode & GOC_ALU_HIGH_A ? GOC_PK_LO_A_HIGH : 0) |
                      (mode & GOC_ALU_HIGH_B ? GOC_PK_LO_B_HIGH : 0) |
                      (mode & GOC_ALU_HIGH_C ? GOC_PK_LO_C_HIGH : 0) |
                      (mode & GOC_ALU_CLAMP ? GOC_PK_CLAMP : 0);
  uint32_t result[32] = {};
  for (int lane = 0; lane < 32; ++lane)
    for (int half = 0; half < (Packed ? 2 : 1); ++half) {
      int sa = half ^ bool(arithmetic_mode & (half ? GOC_PK_HI_A_LOW : GOC_PK_LO_A_HIGH));
      int sb = half ^ bool(arithmetic_mode & (half ? GOC_PK_HI_B_LOW : GOC_PK_LO_B_HIGH));
      int sc = half ^ bool(arithmetic_mode & (half ? GOC_PK_HI_C_LOW : GOC_PK_LO_C_HIGH));
      int64_t x = input<Signed>(a[0][lane], sa), y = input<Signed>(b[0][lane], sb),
              z = input<Signed>(c[0][lane], sc), value;
      if constexpr (Op == goc::Integer16Ternary::Mad) {
        value = x * y + z;
        if (arithmetic_mode & GOC_PK_CLAMP)
          value = std::clamp(value, Signed ? INT64_C(-32768) : INT64_C(0),
                             Signed ? INT64_C(32767) : INT64_C(65535));
      } else if constexpr (Op == goc::Integer16Ternary::Min) {
        value = std::min(std::min(x, y), z);
      } else if constexpr (Op == goc::Integer16Ternary::Max) {
        value = std::max(std::max(x, y), z);
      } else {
        value = std::max(std::min(x, y), std::min(std::max(x, y), z));
      }
      result[lane] |= uint32_t(uint16_t(value)) << (16 * half);
    }
  for (int lane = 0; lane < 32; ++lane)
    if (mask >> lane & 1) {
      if constexpr (Packed) {
        d[0][lane] = result[lane];
      } else {
        int shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
        d[0][lane] = (d[0][lane] & ~(UINT32_C(65535) << shift)) | (result[lane] << shift);
      }
    }
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_pk_mad_i16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  return ternary<true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_rdna4_v_pk_mad_u16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  return ternary<false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_rdna4_v_mad_u16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c) {
  return ternary<false, goc::Integer16Ternary::Mad, false>(flags, exec_mask, instruction_flags, d,
                                                           a, b, c);
}

int goc_rdna4_v_mad_i16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c) {
  return ternary<true, goc::Integer16Ternary::Mad, false>(flags, exec_mask, instruction_flags, d, a,
                                                          b, c);
}

int goc_rdna4_v_min3_u16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c) {
  return ternary<false, goc::Integer16Ternary::Min, false>(flags, exec_mask, instruction_flags, d,
                                                           a, b, c);
}

int goc_rdna4_v_min3_i16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c) {
  return ternary<true, goc::Integer16Ternary::Min, false>(flags, exec_mask, instruction_flags, d, a,
                                                          b, c);
}

int goc_rdna4_v_max3_u16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c) {
  return ternary<false, goc::Integer16Ternary::Max, false>(flags, exec_mask, instruction_flags, d,
                                                           a, b, c);
}

int goc_rdna4_v_max3_i16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c) {
  return ternary<true, goc::Integer16Ternary::Max, false>(flags, exec_mask, instruction_flags, d, a,
                                                          b, c);
}

int goc_rdna4_v_med3_u16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c) {
  return ternary<false, goc::Integer16Ternary::Median, false>(flags, exec_mask, instruction_flags,
                                                              d, a, b, c);
}

int goc_rdna4_v_med3_i16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c) {
  return ternary<true, goc::Integer16Ternary::Median, false>(flags, exec_mask, instruction_flags, d,
                                                             a, b, c);
}

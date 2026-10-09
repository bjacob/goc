// SPDX-License-Identifier: MIT

#include "rdna4_integer16.h"
#include "goc/goc.h"
#include "internal.h"

#include <algorithm>
#include <stdint.h>

namespace {

template <goc::Integer16 Op, bool Signed, bool Packed = true>
int arithmetic(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b) {
  constexpr bool saturating = Op == goc::Integer16::Add || Op == goc::Integer16::Sub;
  const uint32_t known =
      Packed
          ? GOC_PK_LO_A_HIGH | GOC_PK_LO_B_HIGH | GOC_PK_HI_A_LOW | GOC_PK_HI_B_LOW | GOC_PK_CLAMP
          : GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_D | (saturating ? GOC_ALU_CLAMP : 0);
  if (int error = goc::validate(flags, mode & ~known))
    return error;
  if (uint32_t(mask) == 0)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::integer16_x86_64_v3<Op, Signed, Packed>(uint32_t(mask), mode, d[0], a[0], b[0]);
    return GOC_SUCCESS;
  }
#endif
  uint32_t arithmetic_mode = mode;
  if constexpr (!Packed)
    arithmetic_mode = (mode & GOC_ALU_HIGH_A ? GOC_PK_LO_A_HIGH : 0) |
                      (mode & GOC_ALU_HIGH_B ? GOC_PK_LO_B_HIGH : 0) |
                      (mode & GOC_ALU_CLAMP ? GOC_PK_CLAMP : 0);
  uint32_t result[32] = {};
  for (int lane = 0; lane < 32; ++lane) {
    for (int half = 0; half < (Packed ? 2 : 1); ++half) {
      int sa = half ^ bool(arithmetic_mode & (half ? GOC_PK_HI_A_LOW : GOC_PK_LO_A_HIGH));
      int sb = half ^ bool(arithmetic_mode & (half ? GOC_PK_HI_B_LOW : GOC_PK_LO_B_HIGH));
      int64_t x = uint16_t(a[0][lane] >> (16 * sa));
      int64_t y = uint16_t(b[0][lane] >> (16 * sb));
      if constexpr (Signed) {
        x -= (x & 0x8000) ? 65536 : 0;
        y -= (y & 0x8000) ? 65536 : 0;
      }
      int64_t value;
      if constexpr (Op == goc::Integer16::Add)
        value = x + y;
      if constexpr (Op == goc::Integer16::Sub)
        value = x - y;
      if constexpr (Op == goc::Integer16::Min)
        value = std::min(x, y);
      if constexpr (Op == goc::Integer16::Max)
        value = std::max(x, y);
      if constexpr (Op == goc::Integer16::Mul)
        value = x * y;
      if constexpr (Op == goc::Integer16::ShiftLeft)
        value = uint32_t(y) << (uint32_t(x) & 15);

      // Signed Y is already extended to 32 bits. At most 15 right shifts
      // cannot move a zero from above bit 31 into the retained low 16 bits.
      if constexpr (Op == goc::Integer16::ShiftRight)
        value = uint32_t(y) >> (uint32_t(x) & 15);

      if constexpr (Op == goc::Integer16::Add || Op == goc::Integer16::Sub) {
        if (arithmetic_mode & GOC_PK_CLAMP)
          value = std::clamp(value, Signed ? INT64_C(-32768) : INT64_C(0),
                             Signed ? INT64_C(32767) : INT64_C(65535));
      }
      result[lane] |= uint32_t(uint16_t(value)) << (16 * half);
    }
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

int goc_rdna4_v_pk_add_i16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  return arithmetic<goc::Integer16::Add, true>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_pk_sub_i16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  return arithmetic<goc::Integer16::Sub, true>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_pk_add_u16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  return arithmetic<goc::Integer16::Add, false>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_pk_sub_u16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  return arithmetic<goc::Integer16::Sub, false>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_pk_min_i16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  return arithmetic<goc::Integer16::Min, true>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_pk_max_i16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  return arithmetic<goc::Integer16::Max, true>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_pk_min_u16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  return arithmetic<goc::Integer16::Min, false>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_pk_max_u16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  return arithmetic<goc::Integer16::Max, false>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_pk_mul_lo_u16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  return arithmetic<goc::Integer16::Mul, false>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_pk_lshlrev_b16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b) {
  return arithmetic<goc::Integer16::ShiftLeft, false>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_pk_lshrrev_b16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b) {
  return arithmetic<goc::Integer16::ShiftRight, false>(flags, exec_mask, instruction_flags, d, a,
                                                       b);
}

int goc_rdna4_v_pk_ashrrev_i16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b) {
  return arithmetic<goc::Integer16::ShiftRight, true>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_add_nc_i16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  return arithmetic<goc::Integer16::Add, true, false>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_sub_nc_i16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  return arithmetic<goc::Integer16::Sub, true, false>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_add_nc_u16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  return arithmetic<goc::Integer16::Add, false, false>(flags, exec_mask, instruction_flags, d, a,
                                                       b);
}

int goc_rdna4_v_sub_nc_u16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  return arithmetic<goc::Integer16::Sub, false, false>(flags, exec_mask, instruction_flags, d, a,
                                                       b);
}

int goc_rdna4_v_min_i16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  return arithmetic<goc::Integer16::Min, true, false>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_max_i16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  return arithmetic<goc::Integer16::Max, true, false>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_min_u16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  return arithmetic<goc::Integer16::Min, false, false>(flags, exec_mask, instruction_flags, d, a,
                                                       b);
}

int goc_rdna4_v_max_u16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  return arithmetic<goc::Integer16::Max, false, false>(flags, exec_mask, instruction_flags, d, a,
                                                       b);
}

int goc_rdna4_v_mul_lo_u16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  return arithmetic<goc::Integer16::Mul, false, false>(flags, exec_mask, instruction_flags, d, a,
                                                       b);
}

int goc_rdna4_v_lshlrev_b16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a,
                            const uint32_t *const *b) {
  return arithmetic<goc::Integer16::ShiftLeft, false, false>(flags, exec_mask, instruction_flags, d,
                                                             a, b);
}

int goc_rdna4_v_lshrrev_b16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a,
                            const uint32_t *const *b) {
  return arithmetic<goc::Integer16::ShiftRight, false, false>(flags, exec_mask, instruction_flags,
                                                              d, a, b);
}

int goc_rdna4_v_ashrrev_i16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a,
                            const uint32_t *const *b) {
  return arithmetic<goc::Integer16::ShiftRight, true, false>(flags, exec_mask, instruction_flags, d,
                                                             a, b);
}

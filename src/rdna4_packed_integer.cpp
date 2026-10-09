// SPDX-License-Identifier: MIT

#include "rdna4_packed_integer.h"
#include "goc/goc.h"
#include "internal.h"

#include <algorithm>
#include <stdint.h>

namespace {

template <goc::PackedInteger Op, bool Signed>
int arithmetic(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b) {
  const uint32_t known =
      GOC_PK_LO_A_HIGH | GOC_PK_LO_B_HIGH | GOC_PK_HI_A_LOW | GOC_PK_HI_B_LOW | GOC_PK_CLAMP;
  if (int error = goc::validate(flags, mode & ~known))
    return error;
  if (uint32_t(mask) == 0)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::packed_integer_x86_64_v3<Op, Signed>(uint32_t(mask), mode, d[0], a[0], b[0]);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[32] = {};
  for (int lane = 0; lane < 32; ++lane) {
    for (int half = 0; half < 2; ++half) {
      int sa = half ^ bool(mode & (half ? GOC_PK_HI_A_LOW : GOC_PK_LO_A_HIGH));
      int sb = half ^ bool(mode & (half ? GOC_PK_HI_B_LOW : GOC_PK_LO_B_HIGH));
      int64_t x = uint16_t(a[0][lane] >> (16 * sa));
      int64_t y = uint16_t(b[0][lane] >> (16 * sb));
      if constexpr (Signed) {
        x -= (x & 0x8000) ? 65536 : 0;
        y -= (y & 0x8000) ? 65536 : 0;
      }
      int64_t value;
      if constexpr (Op == goc::PackedInteger::Add)
        value = x + y;
      if constexpr (Op == goc::PackedInteger::Sub)
        value = x - y;
      if constexpr (Op == goc::PackedInteger::Min)
        value = std::min(x, y);
      if constexpr (Op == goc::PackedInteger::Max)
        value = std::max(x, y);
      if constexpr (Op == goc::PackedInteger::Mul)
        value = x * y;
      if constexpr (Op == goc::PackedInteger::Add || Op == goc::PackedInteger::Sub) {
        if (mode & GOC_PK_CLAMP)
          value = std::clamp(value, Signed ? INT64_C(-32768) : INT64_C(0),
                             Signed ? INT64_C(32767) : INT64_C(65535));
      }
      result[lane] |= uint32_t(uint16_t(value)) << (16 * half);
    }
  }
  for (int lane = 0; lane < 32; ++lane)
    if (mask >> lane & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_pk_add_i16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  return arithmetic<goc::PackedInteger::Add, true>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_pk_sub_i16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  return arithmetic<goc::PackedInteger::Sub, true>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_pk_add_u16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  return arithmetic<goc::PackedInteger::Add, false>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_pk_sub_u16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  return arithmetic<goc::PackedInteger::Sub, false>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_pk_min_i16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  return arithmetic<goc::PackedInteger::Min, true>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_pk_max_i16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  return arithmetic<goc::PackedInteger::Max, true>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_pk_min_u16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  return arithmetic<goc::PackedInteger::Min, false>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_pk_max_u16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  return arithmetic<goc::PackedInteger::Max, false>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_pk_mul_lo_u16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  return arithmetic<goc::PackedInteger::Mul, false>(flags, exec_mask, instruction_flags, d, a, b);
}

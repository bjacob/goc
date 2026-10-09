// SPDX-License-Identifier: MIT

#include "rdna4_integer_add.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_dpp.h"

#include <algorithm>
#include <stdint.h>

namespace {

template <bool Signed> int64_t value(uint32_t bits) {
  return int64_t(bits) - (Signed && (bits & 0x80000000) ? INT64_C(4294967296) : 0);
}

template <goc::IntegerAdd Op, bool Signed>
int arithmetic(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (mode >> 32) {
    // GFX1201 applies SUBREV's DPP to B before subtracting lane-local A.
    constexpr bool reverse = Op == goc::IntegerAdd::Subrev;
    return goc::execute_dpp(
        flags, mask, mode, reverse ? b : a, [&](uint32_t effective, const uint32_t *const *source) {
          return arithmetic<Op, Signed>(flags, effective, uint32_t(mode), d, reverse ? a : source,
                                        reverse ? source : b, c);
        });
  }

  constexpr bool three = Op == goc::IntegerAdd::Add3;
  const uint32_t known = three ? 0 : GOC_ALU_CLAMP;
  if (int error = goc::validate(flags, mode & ~known))
    return error;
  if (mask == 0)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::integer_add_x86_64_v4<Op, Signed>(mask, mode, d[0], a[0], b[0], three ? c[0] : nullptr);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  // AVX2 saturation beats the baseline path; for wrapping arithmetic, its
  // masked-store overhead outweighs the cheaper vector add/subtract.
  if constexpr (!three) {
    if ((mode & GOC_ALU_CLAMP) && (flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
      goc::integer_add_sat_x86_64_v3<Op, Signed>(mask, d[0], a[0], b[0]);
      return GOC_SUCCESS;
    }
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    int64_t x = value<Signed>(a[0][lane]), y = value<Signed>(b[0][lane]), sum;
    if constexpr (Op == goc::IntegerAdd::Add)
      sum = x + y;
    if constexpr (Op == goc::IntegerAdd::Sub)
      sum = x - y;
    if constexpr (Op == goc::IntegerAdd::Subrev)
      sum = y - x;
    if constexpr (three)
      sum = x + y + c[0][lane];
    if (mode & GOC_ALU_CLAMP)
      sum = std::clamp(sum, Signed ? int64_t(INT32_MIN) : INT64_C(0),
                       Signed ? int64_t(INT32_MAX) : int64_t(UINT32_MAX));
    result[lane] = uint32_t(sum);
  }
  for (int lane = 0; lane < 32; ++lane)
    if (mask >> lane & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_add_nc_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  return arithmetic<goc::IntegerAdd::Add, false>(flags, exec_mask, instruction_flags, d, a, b,
                                                 nullptr);
}

int goc_rdna4_v_sub_nc_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  return arithmetic<goc::IntegerAdd::Sub, false>(flags, exec_mask, instruction_flags, d, a, b,
                                                 nullptr);
}

int goc_rdna4_v_subrev_nc_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  return arithmetic<goc::IntegerAdd::Subrev, false>(flags, exec_mask, instruction_flags, d, a, b,
                                                    nullptr);
}

int goc_rdna4_v_add_nc_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  return arithmetic<goc::IntegerAdd::Add, true>(flags, exec_mask, instruction_flags, d, a, b,
                                                nullptr);
}

int goc_rdna4_v_sub_nc_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  return arithmetic<goc::IntegerAdd::Sub, true>(flags, exec_mask, instruction_flags, d, a, b,
                                                nullptr);
}

int goc_rdna4_v_add3_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c) {
  return arithmetic<goc::IntegerAdd::Add3, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

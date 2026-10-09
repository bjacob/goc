// SPDX-License-Identifier: MIT
// 24-bit truncation/sign extension follow rocjitsu shared/simd_glue.h.

#include "rdna4_integer_mul.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_dpp.h"

#include <algorithm>
#include <stdint.h>

namespace {

template <int Bits> int64_t signed_value(uint32_t value) {
  if constexpr (Bits == 24)
    value &= 0x00ffffff;
  return int64_t(value) - ((value & (UINT32_C(1) << (Bits - 1))) ? (INT64_C(1) << Bits) : 0);
}

template <int Bits, bool Signed, bool High>
int multiply(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
             const uint32_t *const *a, const uint32_t *const *b) {
  if (mode >> 32) {
    if constexpr (Bits == 24) {
      return goc::execute_dpp(
          flags, mask, mode, a, [&](uint32_t effective, const uint32_t *const *source) {
            return multiply<Bits, Signed, High>(flags, effective, uint32_t(mode), d, source, b);
          });
    } else {
      return GOC_ERROR_INVALID_FLAGS;
    }
  }

  const uint32_t known = Bits == 24 && !High ? GOC_ALU_CLAMP : 0;
  if (int error = goc::validate(flags, mode & ~known))
    return error;
  if (mask == 0)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::integer_mul_x86_64_v4<Bits, Signed, High>(mask, mode, d[0], a[0], b[0]);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::integer_mul_x86_64_v3<Bits, Signed, High>(mask, mode, d[0], a[0], b[0]);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    uint64_t product;
    if constexpr (Signed) {
      int64_t value = signed_value<Bits>(a[0][lane]) * signed_value<Bits>(b[0][lane]);
      if (mode & GOC_ALU_CLAMP)
        value = std::clamp(value, int64_t(INT32_MIN), int64_t(INT32_MAX));
      product = uint64_t(value);
    } else {
      const uint32_t input_mask = Bits == 24 ? UINT32_C(0x00ffffff) : UINT32_MAX;
      product = uint64_t(a[0][lane] & input_mask) * (b[0][lane] & input_mask);
      if (mode & GOC_ALU_CLAMP)
        product = std::min(product, uint64_t(UINT32_MAX));
    }
    result[lane] = uint32_t(High ? product >> 32 : product);
  }
  for (int lane = 0; lane < 32; ++lane)
    if (mask >> lane & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_mul_lo_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return multiply<32, false, false>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_mul_hi_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return multiply<32, false, true>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_mul_hi_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return multiply<32, true, true>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_mul_i32_i24(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a,
                            const uint32_t *const *b) {
  return multiply<24, true, false>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_mul_hi_i32_i24(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b) {
  return multiply<24, true, true>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_mul_u32_u24(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a,
                            const uint32_t *const *b) {
  return multiply<24, false, false>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_mul_hi_u32_u24(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b) {
  return multiply<24, false, true>(flags, exec_mask, instruction_flags, d, a, b);
}

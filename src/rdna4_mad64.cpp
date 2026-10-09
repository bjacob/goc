// SPDX-License-Identifier: MIT

#include "rdna4_mad64.h"
#include "goc/goc.h"
#include "internal.h"

#include <stdint.h>

namespace {

template <bool Signed>
int run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d, uint32_t *carry,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (int error = goc::validate(flags, mode & ~GOC_ALU_CLAMP, true))
    return error;
  if (!exec_mask) {
    *carry = 0;
    return GOC_SUCCESS;
  }
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    *carry = goc::mad64_x86_64_v4<Signed>(exec_mask, mode, d, a, b, c);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    *carry = goc::mad64_x86_64_v3<Signed>(exec_mask, mode, d, a, b, c);
    return GOC_SUCCESS;
  }
#endif
  uint64_t result[32];
  uint32_t output_carry = 0;
  for (unsigned lane = 0; lane < 32; ++lane) {
    uint32_t av = a[0][lane], bv = b[0][lane];
    uint64_t cv = c[0][lane] | (uint64_t(c[1][lane]) << 32), product;
    if constexpr (Signed) {
      int64_t sa = int64_t(av) - (int64_t(av >> 31) << 32),
              sb = int64_t(bv) - (int64_t(bv >> 31) << 32);
      product = uint64_t(sa * sb);
    } else
      product = uint64_t(av) * bv;
    uint64_t sum = product + cv;
    bool overflow, co;
    if constexpr (Signed) {
      overflow = (((product ^ sum) & (cv ^ sum)) >> 63) != 0;
      co = bool(sum >> 63) ^ overflow;
      if ((mode & GOC_ALU_CLAMP) && overflow)
        sum = (product >> 63) ? 1ULL << 63 : UINT64_MAX >> 1;
    } else {
      overflow = sum < product;
      co = overflow;
      if ((mode & GOC_ALU_CLAMP) && overflow)
        sum = UINT64_MAX;
    }
    result[lane] = sum;
    output_carry |= uint32_t(co) << lane;
  }
  for (unsigned reg = 0; reg < 2; ++reg)
    for (unsigned lane = 0; lane < 32; ++lane)
      if ((exec_mask >> lane) & 1)
        d[reg][lane] = uint32_t(result[lane] >> (32 * reg));
  *carry = output_carry & exec_mask;
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_mad_co_u64_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, uint32_t *carry, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<false>(flags, exec_mask, instruction_flags, d, carry, a, b, c);
}

int goc_rdna4_v_mad_co_i64_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, uint32_t *carry, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<true>(flags, exec_mask, instruction_flags, d, carry, a, b, c);
}

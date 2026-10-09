// SPDX-License-Identifier: MIT

// Source sign-bit modifiers follow rocjitsu's conditional-selection helpers.

#include "rdna4_cndmask.h"
#include "goc/goc.h"
#include "internal.h"

#include <stdint.h>

namespace {

template <bool Half>
int run(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d, const uint32_t *const *a,
        const uint32_t *const *b, uint32_t condition) {
  const uint32_t known = GOC_ALU_ABS_A | GOC_ALU_ABS_B | GOC_ALU_NEG_A | GOC_ALU_NEG_B |
                         (Half ? GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_D : 0);
  if (int error = goc::validate(flags, mode & ~known))
    return error;
  if (!uint32_t(mask))
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::cndmask_x86_64_v4<Half>(uint32_t(mask), mode, d[0], a[0], b[0], condition);
    return GOC_SUCCESS;
  }
#endif
  // AVX2 candidates did not provide a substantial gain over the portable path.

  const uint32_t sign = Half ? 0x8000 : 0x80000000;
  unsigned sa = Half && (mode & GOC_ALU_HIGH_A) ? 16 : 0;
  unsigned sb = Half && (mode & GOC_ALU_HIGH_B) ? 16 : 0;
  unsigned sd = Half && (mode & GOC_ALU_HIGH_D) ? 16 : 0;
  uint32_t clear_a = mode & GOC_ALU_ABS_A ? ~sign : UINT32_MAX;
  uint32_t clear_b = mode & GOC_ALU_ABS_B ? ~sign : UINT32_MAX;
  uint32_t flip_a = mode & GOC_ALU_NEG_A ? sign : 0;
  uint32_t flip_b = mode & GOC_ALU_NEG_B ? sign : 0;
  uint32_t result[32];
  for (unsigned lane = 0; lane < 32; ++lane) {
    uint32_t x = ((a[0][lane] >> sa) & clear_a) ^ flip_a;
    uint32_t y = ((b[0][lane] >> sb) & clear_b) ^ flip_b;
    uint32_t value = ((condition >> lane) & 1) ? y : x;
    if constexpr (Half)
      value = (d[0][lane] & ~(65535u << sd)) | ((value & 65535) << sd);
    result[lane] = value;
  }
  for (unsigned lane = 0; lane < 32; ++lane)
    if ((mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_cndmask_b32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            uint32_t condition) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<false>(flags, exec_mask, instruction_flags, d, a, b, condition);
}

int goc_rdna4_v_cndmask_b16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            uint32_t condition) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<true>(flags, exec_mask, instruction_flags, d, a, b, condition);
}

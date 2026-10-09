// SPDX-License-Identifier: MIT

#include "rdna4_packed_mad.h"
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

template <bool Signed>
int mad(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d, const uint32_t *const *a,
        const uint32_t *const *b, const uint32_t *const *c) {
  const uint32_t known = GOC_PK_LO_A_HIGH | GOC_PK_LO_B_HIGH | GOC_PK_LO_C_HIGH | GOC_PK_HI_A_LOW |
                         GOC_PK_HI_B_LOW | GOC_PK_HI_C_LOW | GOC_PK_CLAMP;
  if (int error = goc::validate(flags, mode & ~known))
    return error;
  if (uint32_t(mask) == 0)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::packed_mad_x86_64_v3<Signed>(uint32_t(mask), mode, d[0], a[0], b[0], c[0]);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[32] = {};
  for (int lane = 0; lane < 32; ++lane)
    for (int half = 0; half < 2; ++half) {
      int sa = half ^ bool(mode & (half ? GOC_PK_HI_A_LOW : GOC_PK_LO_A_HIGH));
      int sb = half ^ bool(mode & (half ? GOC_PK_HI_B_LOW : GOC_PK_LO_B_HIGH));
      int sc = half ^ bool(mode & (half ? GOC_PK_HI_C_LOW : GOC_PK_LO_C_HIGH));
      int64_t value = input<Signed>(a[0][lane], sa) * input<Signed>(b[0][lane], sb) +
                      input<Signed>(c[0][lane], sc);
      if (mode & GOC_PK_CLAMP)
        value = std::clamp(value, Signed ? INT64_C(-32768) : INT64_C(0),
                           Signed ? INT64_C(32767) : INT64_C(65535));
      result[lane] |= uint32_t(uint16_t(value)) << (16 * half);
    }
  for (int lane = 0; lane < 32; ++lane)
    if (mask >> lane & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_pk_mad_i16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  return mad<true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_rdna4_v_pk_mad_u16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  return mad<false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

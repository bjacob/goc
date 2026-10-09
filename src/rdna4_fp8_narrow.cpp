// SPDX-License-Identifier: MIT

#include "rdna4_fp8_narrow.h"
#include "goc/goc.h"
#include "internal.h"

#include <stdint.h>

namespace {

template <bool Bf8, bool Stochastic>
int convert(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b) {
  const uint32_t known =
      GOC_ALU_ABS_A | GOC_ALU_NEG_A |
      (Stochastic ? GOC_CVT_BYTE_3 : GOC_ALU_ABS_B | GOC_ALU_NEG_B | GOC_ALU_HIGH_D);
  if (int error = goc::validate(flags, mode & ~known))
    return error;
  if (!uint32_t(mask))
    return GOC_SUCCESS;
  bool saturate = flags & GOC_FP16_OVFL;
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::fp8_narrow_x86_64_v4<Bf8, Stochastic>(uint32_t(mask), mode, saturate, d[0], a[0], b[0]);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::fp8_narrow_x86_64_v3<Bf8, Stochastic>(uint32_t(mask), mode, saturate, d[0], a[0], b[0]);
    return GOC_SUCCESS;
  }
#endif
  unsigned shift = Stochastic ? ((mode >> 16) & 3) * 8 : mode & GOC_ALU_HIGH_D ? 16 : 0;
  uint32_t selected = (Stochastic ? 255u : 65535u) << shift;
  uint32_t result[32];
  for (unsigned lane = 0; lane < 32; ++lane) {
    uint32_t av = a[0][lane], bv = b[0][lane];
    av &= mode & GOC_ALU_ABS_A ? 0x7fffffff : UINT32_MAX;
    av ^= mode & GOC_ALU_NEG_A ? 0x80000000 : 0;
    uint32_t value = goc::narrow_fp8<Bf8, Stochastic>(av, bv, saturate);
    if constexpr (!Stochastic) {
      bv &= mode & GOC_ALU_ABS_B ? 0x7fffffff : UINT32_MAX;
      bv ^= mode & GOC_ALU_NEG_B ? 0x80000000 : 0;
      value |= goc::narrow_fp8<Bf8, false>(bv, 0, saturate) << 8;
    }
    result[lane] = (d[0][lane] & ~selected) | (value << shift);
  }
  for (unsigned lane = 0; lane < 32; ++lane)
    if ((mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_cvt_pk_fp8_f32(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                               const uint32_t *const *a, const uint32_t *const *b) {
  return convert<false, false>(flags, mask, mode, d, a, b);
}

int goc_rdna4_v_cvt_pk_bf8_f32(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                               const uint32_t *const *a, const uint32_t *const *b) {
  return convert<true, false>(flags, mask, mode, d, a, b);
}

int goc_rdna4_v_cvt_sr_fp8_f32(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                               const uint32_t *const *a, const uint32_t *const *b) {
  return convert<false, true>(flags, mask, mode, d, a, b);
}

int goc_rdna4_v_cvt_sr_bf8_f32(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                               const uint32_t *const *a, const uint32_t *const *b) {
  return convert<true, true>(flags, mask, mode, d, a, b);
}

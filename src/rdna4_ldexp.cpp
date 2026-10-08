// SPDX-License-Identifier: MIT

#include "rdna4_ldexp.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_alu.h"
#include "rdna4_fp64.h"

#include <cmath>
#include <cstring>
#include <stdint.h>

namespace {

template <bool Fp64>
int ldexp(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
          const uint32_t *const *a, const uint32_t *const *b) {
  const uint32_t known = GOC_ALU_NEG_A | GOC_ALU_ABS_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP;
  if (int error = goc::validate(flags, mode & ~known))
    return error;
  if (uint32_t(mask) == 0)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::ldexp_x86_64_v4(Fp64, uint32_t(mask), mode, d, a, b[0]);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::ldexp_x86_64_v3(Fp64, uint32_t(mask), mode, d, a, b[0]);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[Fp64 ? 2 : 1][32];
  for (int lane = 0; lane < 32; ++lane) {
    int32_t exponent;
    std::memcpy(&exponent, b[0] + lane, sizeof(exponent));
    if constexpr (Fp64) {
      double value = std::ldexp(goc::fp64_input(a, lane, mode), exponent);
      uint64_t bits = goc::double_bits(goc::fp64_output(value, mode));
      result[0][lane] = uint32_t(bits);
      result[1][lane] = uint32_t(bits >> 32);
    } else {
      float value = std::ldexp(goc::alu_input(a[0][lane], mode), exponent);
      result[0][lane] = goc::as_bits(goc::alu_output(value, mode));
    }
  }
  for (int reg = 0; reg < (Fp64 ? 2 : 1); ++reg)
    for (int lane = 0; lane < 32; ++lane)
      if ((mask >> lane) & 1)
        d[reg][lane] = result[reg][lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_ldexp_f32(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b) {
  return ldexp<false>(flags, mask, mode, d, a, b);
}

int goc_rdna4_v_ldexp_f64(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b) {
  return ldexp<true>(flags, mask, mode, d, a, b);
}

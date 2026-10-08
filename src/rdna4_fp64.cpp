// SPDX-License-Identifier: MIT

#include "rdna4_fp64.h"
#include "goc/goc.h"
#include "internal.h"

#include <cmath>
#include <stdint.h>

namespace {

template <goc::Fp64 Op>
int arithmetic(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  const uint32_t known = Op == goc::Fp64::Fma
                             ? 0x1ff
                             : GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_ABS_B | GOC_ALU_NEG_B |
                                   GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP;
  if (int error = goc::validate(flags, mode & ~known))
    return error;
  if (uint32_t(mask) == 0)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::fp64_x86_64_v3(Op, uint32_t(mask), mode, d, a, b, c);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[2][32];
  for (int lane = 0; lane < 32; ++lane) {
    double x = goc::fp64_input(a, lane, mode), y = goc::fp64_input(b, lane, mode >> 1);
    double value;
    if constexpr (Op == goc::Fp64::Add)
      value = x + y;
    if constexpr (Op == goc::Fp64::Mul)
      value = x * y;
    if constexpr (Op == goc::Fp64::Fma)
      value = std::fma(x, y, goc::fp64_input(c, lane, mode >> 2));
    uint64_t bits = goc::double_bits(goc::fp64_output(value, mode));
    result[0][lane] = uint32_t(bits);
    result[1][lane] = uint32_t(bits >> 32);
  }
  // Stage both halves before any destination write, including cross-half aliases.
  for (int reg = 0; reg < 2; ++reg)
    for (int lane = 0; lane < 32; ++lane)
      if ((mask >> lane) & 1)
        d[reg][lane] = result[reg][lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_add_f64(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b) {
  return arithmetic<goc::Fp64::Add>(flags, mask, mode, d, a, b, nullptr);
}

int goc_rdna4_v_mul_f64(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b) {
  return arithmetic<goc::Fp64::Mul>(flags, mask, mode, d, a, b, nullptr);
}

int goc_rdna4_v_fma_f64(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c) {
  return arithmetic<goc::Fp64::Fma>(flags, mask, mode, d, a, b, c);
}

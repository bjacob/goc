// SPDX-License-Identifier: MIT

#include "rdna4_frexp_exp.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_fp64.h"

#include <cmath>
#include <stdint.h>

namespace {

template <bool Fp64>
int frexp_exp(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
              const uint32_t *const *a) {
  const uint32_t known = GOC_ALU_NEG_A | GOC_ALU_ABS_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP;
  if (int error = goc::validate(flags, mode & ~known))
    return error;
  if (uint32_t(mask) == 0)
    return GOC_SUCCESS;
  // Sign modifiers do not affect the exponent. Integer results ignore OMOD;
  // the exponent always fits in int32_t, so integer CLAMP has no effect.
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::frexp_exp_x86_64_v3(Fp64, uint32_t(mask), d[0], a);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    // Match rocjitsu's frexp exponent model, including zero for NaNs/infinities.
    double value;
    if constexpr (Fp64)
      value = goc::as_double(uint64_t(a[0][lane]) | (uint64_t(a[1][lane]) << 32));
    else
      value = goc::as_float(a[0][lane]);
    int exponent = 0;
    if (value != 0 && std::isfinite(value))
      std::frexp(value, &exponent);
    result[lane] = uint32_t(exponent);
  }
  for (int lane = 0; lane < 32; ++lane)
    if ((mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_frexp_exp_i32_f32(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                                  const uint32_t *const *a) {
  return frexp_exp<false>(flags, mask, mode, d, a);
}

int goc_rdna4_v_frexp_exp_i32_f64(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                                  const uint32_t *const *a) {
  return frexp_exp<true>(flags, mask, mode, d, a);
}

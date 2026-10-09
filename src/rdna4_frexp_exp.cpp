// SPDX-License-Identifier: MIT

#include "rdna4_frexp_exp.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_dpp.h"
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
    int exponent = 0;
    if constexpr (Fp64) {
      double value = goc::as_double(uint64_t(a[0][lane]) | (uint64_t(a[1][lane]) << 32));
      if (value != 0 && std::isfinite(value))
        std::frexp(value, &exponent);
    } else {
      uint32_t magnitude = a[0][lane] & 0x7fffffffu;
      unsigned field = magnitude >> 23;
      if (field && field != 255)
        exponent = int(field) - 126;
      else if (!field && magnitude) {
        exponent = -149;
        while (magnitude) {
          ++exponent;
          magnitude >>= 1;
        }
      }
    }
    result[lane] = uint32_t(exponent);
  }
  for (int lane = 0; lane < 32; ++lane)
    if ((mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_frexp_exp_i32_f32(uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                                  const uint32_t *const *a) {
  if (mode >> 32)
    return goc::execute_dpp(flags, mask, mode, a,
                            [&](uint32_t effective, const uint32_t *const *source) {
                              return frexp_exp<false>(flags, effective, uint32_t(mode), d, source);
                            });
  return frexp_exp<false>(flags, mask, mode, d, a);
}

int goc_rdna4_v_frexp_exp_i32_f64(uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                                  const uint32_t *const *a) {
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return frexp_exp<true>(flags, mask, mode, d, a);
}

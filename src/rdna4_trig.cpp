// SPDX-License-Identifier: MIT

#include "rdna4_trig.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_dpp.h"
#include "rdna4_trig_model.h"

#include <stdint.h>

namespace {

// Scale a finite trig result by an exact power of two with RNE underflow.
// Input magnitude is at most one; no multiplication can overflow.
uint32_t output(uint32_t bits, uint32_t mode) {
  uint32_t magnitude = bits & 0x7fffffff;
  if (magnitude < 0x7f800000) {
    unsigned scale = (mode >> 6) & 3;
    if (scale == 3) {
      magnitude =
          magnitude < 0x1000000 ? (magnitude >> 1) + ((magnitude & 3) == 3) : magnitude - 0x800000;
    } else {
      for (unsigned i = 0; i < scale; ++i)
        magnitude = magnitude < 0x800000 ? magnitude * 2 : magnitude + 0x800000;
    }
    bits = (bits & 0x80000000) | magnitude;
  }
  if (mode & GOC_ALU_CLAMP) {
    if ((bits >> 31) || magnitude > 0x7f800000)
      return 0;
    if (magnitude > 0x3f800000)
      return 0x3f800000;
  }
  if ((mode & GOC_ALU_OMOD_HALF) && (bits & 0x7fffffff) < 0x800000)
    return 0;
  return bits;
}

template <bool Cosine>
int trig(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
         const uint32_t *const *a) {
  if (mode >> 32)
    return goc::execute_dpp(flags, mask, mode, a,
                            [&](uint32_t effective, const uint32_t *const *source) {
                              return trig<Cosine>(flags, effective, uint32_t(mode), d, source);
                            });
  const uint32_t known = GOC_ALU_NEG_A | GOC_ALU_ABS_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP;
  if (int error = goc::validate(flags, mode & ~known, true))
    return error;
  if (!mask)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3 &&
      (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_EXACT_EMPIRICAL) {
    goc::trig_x86_64_v3(Cosine, mask, mode, d[0], a[0]);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    uint32_t bits = a[0][lane];
    if (mode & GOC_ALU_ABS_A)
      bits &= 0x7fffffff;
    if (mode & GOC_ALU_NEG_A)
      bits ^= 0x80000000;
    result[lane] = output(goc::trig::evaluate(bits, Cosine), mode);
  }
  for (int lane = 0; lane < 32; ++lane)
    if ((mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_sin_f32(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
                        const uint32_t *const *a) {
  return trig<false>(flags, mask, mode, d, a);
}

int goc_rdna4_v_cos_f32(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
                        const uint32_t *const *a) {
  return trig<true>(flags, mask, mode, d, a);
}

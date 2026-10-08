// SPDX-License-Identifier: MIT

#include "float_formats.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_simd.h"

#include <cmath>
#include <stdint.h>

namespace {

template <bool Bf8A, bool Bf8B>
int dot(uint64_t flags, uint64_t mask, uint32_t modifiers, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  const uint32_t known = GOC_DOT_NEG_C | GOC_DOT_ABS_C;
  if (int error = goc::validate(flags, modifiers & ~known))
    return error;
  if (uint32_t(mask) == 0)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::fp8_dot_x86_64_v3(Bf8A, Bf8B, uint32_t(mask), modifiers, d[0], a[0], b[0], c[0]);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    uint32_t acc_bits = c[0][lane];
    if (modifiers & GOC_DOT_ABS_C)
      acc_bits &= 0x7fffffff;
    if (modifiers & GOC_DOT_NEG_C)
      acc_bits ^= 0x80000000;
    float acc = goc::as_float(acc_bits);
    for (int shift = 0; shift < 32; shift += 8)
      acc = std::fma(goc::fp8_to_float<Bf8A>(uint8_t(a[0][lane] >> shift)),
                     goc::fp8_to_float<Bf8B>(uint8_t(b[0][lane] >> shift)), acc);
    result[lane] = goc::as_bits(acc);
  }
  for (int lane = 0; lane < 32; ++lane)
    if ((mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_dot4_f32_fp8_fp8(uint64_t flags, uint64_t mask, uint32_t modifiers,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c) {
  return dot<false, false>(flags, mask, modifiers, d, a, b, c);
}

int goc_rdna4_v_dot4_f32_fp8_bf8(uint64_t flags, uint64_t mask, uint32_t modifiers,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c) {
  return dot<false, true>(flags, mask, modifiers, d, a, b, c);
}

int goc_rdna4_v_dot4_f32_bf8_fp8(uint64_t flags, uint64_t mask, uint32_t modifiers,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c) {
  return dot<true, false>(flags, mask, modifiers, d, a, b, c);
}

int goc_rdna4_v_dot4_f32_bf8_bf8(uint64_t flags, uint64_t mask, uint32_t modifiers,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c) {
  return dot<true, true>(flags, mask, modifiers, d, a, b, c);
}

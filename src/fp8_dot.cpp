// SPDX-License-Identifier: MIT

#include "float_formats.h"
#include "goc/goc.h"
#include "internal.h"
#include "simd.h"

#include <cmath>
#include <stdint.h>

namespace {

template <bool Bf8A, bool Bf8B>
int dot(uint64_t flags, uint32_t exec_mask, uint32_t modifiers, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  const uint32_t known = GOC_DOT_NEG_C | GOC_DOT_ABS_C;
  if (int error = goc::validate(flags, modifiers & ~known))
    return error;
  if (exec_mask == 0)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::fp8_dot_x86_64_v3(Bf8A, Bf8B, exec_mask, modifiers, d[0], a[0], b[0], c[0]);
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
    if ((exec_mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

template <bool Bf8A, bool Bf8B>
int dot_wave64(uint64_t flags, uint64_t exec_mask, uint32_t mode, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (int error = dot<Bf8A, Bf8B>(flags, 0, mode, nullptr, nullptr, nullptr, nullptr))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
  for (unsigned half = 0; half < 2; ++half) {
    uint32_t half_mode = mode;
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *av = a[0] + half * 32, *bv = b[0] + half * 32, *cv = c[0] + half * 32;
    if (int error = dot<Bf8A, Bf8B>(flags, uint32_t(exec_mask >> (half * 32)), half_mode, &dest,
                                    &av, &bv, &cv))
      return error;
  }
  return GOC_SUCCESS;
}

} // namespace

int goc_v_dot4_f32_fp8_fp8(uint64_t flags, uint32_t exec_mask, uint64_t modifiers,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  if (modifiers >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return dot<false, false>(flags, exec_mask, modifiers, d, a, b, c);
}

int goc_v_dot4_f32_fp8_bf8(uint64_t flags, uint32_t exec_mask, uint64_t modifiers,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  if (modifiers >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return dot<false, true>(flags, exec_mask, modifiers, d, a, b, c);
}

int goc_v_dot4_f32_bf8_fp8(uint64_t flags, uint32_t exec_mask, uint64_t modifiers,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  if (modifiers >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return dot<true, false>(flags, exec_mask, modifiers, d, a, b, c);
}

int goc_v_dot4_f32_bf8_bf8(uint64_t flags, uint32_t exec_mask, uint64_t modifiers,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  if (modifiers >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return dot<true, true>(flags, exec_mask, modifiers, d, a, b, c);
}

int goc_v_dot4_f32_fp8_fp8_wave64(uint64_t flags, uint64_t exec_mask, uint64_t modifiers,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c) {
  if (modifiers >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return dot_wave64<false, false>(flags, exec_mask, modifiers, d, a, b, c);
}

int goc_v_dot4_f32_fp8_bf8_wave64(uint64_t flags, uint64_t exec_mask, uint64_t modifiers,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c) {
  if (modifiers >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return dot_wave64<false, true>(flags, exec_mask, modifiers, d, a, b, c);
}

int goc_v_dot4_f32_bf8_fp8_wave64(uint64_t flags, uint64_t exec_mask, uint64_t modifiers,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c) {
  if (modifiers >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return dot_wave64<true, false>(flags, exec_mask, modifiers, d, a, b, c);
}

int goc_v_dot4_f32_bf8_bf8_wave64(uint64_t flags, uint64_t exec_mask, uint64_t modifiers,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c) {
  if (modifiers >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return dot_wave64<true, true>(flags, exec_mask, modifiers, d, a, b, c);
}

// SPDX-License-Identifier: MIT
// True16 DOT2 association and denormal policy follow rocjitsu shared/fp_mode.h.

#include "float_formats.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_dpp.h"
#include "rdna4_simd.h"

#include <stdint.h>

namespace {

template <bool Bf16> float input(uint16_t bits, uint32_t mode) {
  if (mode & GOC_ALU_ABS_A)
    bits &= 0x7fff;
  if (mode & GOC_ALU_NEG_A)
    bits ^= 0x8000;
  if constexpr (Bf16) {
    if (!(bits & 0x7f80))
      bits &= 0x8000;
    return goc::bf16_to_float(bits);
  } else
    return goc::f16_to_float(bits);
}

template <bool Bf16>
int dot(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d, const uint32_t *const *a,
        const uint32_t *const *b, const uint32_t *const *c) {
  if (mode >> 32)
    return goc::execute_dpp(flags, mask, mode, a,
                            [&](uint32_t effective, const uint32_t *const *source) {
                              return dot<Bf16>(flags, effective, uint32_t(mode), d, source, b, c);
                            });
  // Six ABS/NEG bits, plus C and D half selectors.
  if (int error = goc::validate(flags, mode & ~(63U | GOC_ALU_HIGH_C | GOC_ALU_HIGH_D)))
    return error;
  if (mask == 0)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::half_dot_x86_64_v3(Bf16, bool(flags & GOC_FP16_OVFL), mask, mode, d[0], a[0], b[0], c[0]);
    return GOC_SUCCESS;
  }
#endif
  int c_shift = mode & GOC_ALU_HIGH_C ? 16 : 0;
  int d_shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  uint16_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    float p0 =
        input<Bf16>(uint16_t(a[0][lane]), mode) * input<Bf16>(uint16_t(b[0][lane]), mode >> 1);
    float p1 = input<Bf16>(uint16_t(a[0][lane] >> 16), mode) *
               input<Bf16>(uint16_t(b[0][lane] >> 16), mode >> 1);
    float value = (p0 + p1) + input<Bf16>(uint16_t(c[0][lane] >> c_shift), mode >> 2);
    if constexpr (Bf16) {
      result[lane] = goc::float_to_bf16(value);
      if (!(result[lane] & 0x7f80))
        result[lane] &= 0x8000;
    } else
      result[lane] = goc::float_to_f16(value, flags & GOC_FP16_OVFL);
  }
  for (int lane = 0; lane < 32; ++lane)
    if ((mask >> lane) & 1)
      d[0][lane] = (d[0][lane] & ~(0xffffU << d_shift)) | (uint32_t(result[lane]) << d_shift);
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_dot2_f16_f16(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c) {
  return dot<false>(flags, mask, mode, d, a, b, c);
}

int goc_rdna4_v_dot2_bf16_bf16(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               const uint32_t *const *c) {
  return dot<true>(flags, mask, mode, d, a, b, c);
}

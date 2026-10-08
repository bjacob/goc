// SPDX-License-Identifier: MIT

#include "rdna4_dot.h"
#include "float_formats.h"
#include "goc/goc.h"
#include "internal.h"

#include <array>
#include <cmath>
#include <stdint.h>

namespace {

template <bool Bf16>
int dot(uint64_t flags, uint64_t mask, uint32_t instruction_flags, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;

  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane)
    if ((mask >> lane) & 1) {
      std::array<uint16_t, 2> left = {uint16_t(a[0][lane]), uint16_t(a[0][lane] >> 16)};
      std::array<uint16_t, 2> right = {uint16_t(b[0][lane]), uint16_t(b[0][lane] >> 16)};
      if ((flags & GOC_SEMANTICS_MASK) == GOC_SEMANTICS_EXACT_EMPIRICAL)
        result[lane] = goc::gfx12_dot_bits<Bf16, 2>(left, right, c[0][lane]);
      else {
        float acc = goc::as_float(c[0][lane]);
        for (int j = 0; j < 2; ++j) {
          float x = Bf16 ? goc::bf16_to_float(left[j]) : goc::f16_to_float(left[j]);
          float y = Bf16 ? goc::bf16_to_float(right[j]) : goc::f16_to_float(right[j]);
          acc = std::fma(x, y, acc);
        }
        result[lane] = goc::as_bits(acc);
      }
    }

  for (int lane = 0; lane < 32; ++lane)
    if ((mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_dot2_f32_f16(uint64_t flags, uint64_t mask, uint32_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c) {
  return dot<false>(flags, mask, instruction_flags, d, a, b, c);
}

int goc_rdna4_v_dot2_f32_bf16(uint64_t flags, uint64_t mask, uint32_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c) {
  return dot<true>(flags, mask, instruction_flags, d, a, b, c);
}

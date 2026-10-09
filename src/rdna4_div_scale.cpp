// SPDX-License-Identifier: MIT

#include "rdna4_div_scale.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_division.h"

#include <stdint.h>

namespace {

template <unsigned Width>
int run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d, uint32_t *condition,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  const uint32_t known =
      GOC_ALU_NEG_A | GOC_ALU_NEG_B | GOC_ALU_NEG_C | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP;
  if (int error = goc::validate(flags, mode & ~known, true))
    return error;
  uint32_t mask = exec_mask;
  if (!mask) {
    *condition = 0;
    return GOC_SUCCESS;
  }
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    *condition = goc::div_scale_x86_64_v4<Width>(mask, mode, d, a, b, c);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    *condition = goc::div_scale_x86_64_v3<Width>(mask, mode, d, a, b, c);
    return GOC_SUCCESS;
  }
#endif
  using T = typename goc::DivisionFormat<Width>::Bits;
  T result[32];
  uint32_t output_mask = 0;
  for (unsigned lane = 0; lane < 32; ++lane) {
    T av = a[0][lane], bv = b[0][lane], cv = c[0][lane];
    if constexpr (Width == 64) {
      av |= T(a[1][lane]) << 32;
      bv |= T(b[1][lane]) << 32;
      cv |= T(c[1][lane]) << 32;
    }
    auto scaled = goc::division_scale<Width>(av, bv, cv, mode);
    result[lane] = scaled.value;
    output_mask |= uint32_t(scaled.post_scale) << lane;
  }
  for (unsigned reg = 0; reg < (Width == 64 ? 2u : 1u); ++reg)
    for (unsigned lane = 0; lane < 32; ++lane)
      if ((mask >> lane) & 1)
        d[reg][lane] = uint32_t(result[lane] >> (Width == 64 ? 32 * reg : 0));
  *condition = output_mask & mask;
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_div_scale_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, uint32_t *condition, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32>(flags, exec_mask, instruction_flags, d, condition, a, b, c);
}

int goc_rdna4_v_div_scale_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, uint32_t *condition, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64>(flags, exec_mask, instruction_flags, d, condition, a, b, c);
}

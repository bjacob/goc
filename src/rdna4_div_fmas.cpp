// SPDX-License-Identifier: MIT

#include "rdna4_div_fmas.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_division.h"

#include <stdint.h>

namespace {

template <unsigned Width>
int run(uint64_t flags, uint64_t exec_mask, uint32_t mode, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
        uint32_t condition) {
  if (int error = goc::validate(flags, mode & ~511u, true))
    return error;
  uint32_t mask = uint32_t(exec_mask);
  if (!mask)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::div_fmas_x86_64_v4<Width>(mask, mode, d, a, b, c, condition);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::div_fmas_x86_64_v3<Width>(mask, mode, d, a, b, c, condition);
    return GOC_SUCCESS;
  }
#endif
  using T = typename goc::DivisionFormat<Width>::Bits;
  T result[32];
  for (unsigned lane = 0; lane < 32; ++lane) {
    T av = a[0][lane], bv = b[0][lane], cv = c[0][lane];
    if constexpr (Width == 64) {
      av |= T(a[1][lane]) << 32;
      bv |= T(b[1][lane]) << 32;
      cv |= T(c[1][lane]) << 32;
    }
    result[lane] = goc::division_fmas<Width>(av, bv, cv, (condition >> lane) & 1, mode);
  }
  for (unsigned reg = 0; reg < (Width == 64 ? 2u : 1u); ++reg)
    for (unsigned lane = 0; lane < 32; ++lane)
      if ((mask >> lane) & 1)
        d[reg][lane] = uint32_t(result[lane] >> (Width == 64 ? 32 * reg : 0));
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_div_fmas_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c, uint32_t condition) {
  return run<32>(flags, exec_mask, instruction_flags, d, a, b, c, condition);
}

int goc_rdna4_v_div_fmas_f64(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c, uint32_t condition) {
  return run<64>(flags, exec_mask, instruction_flags, d, a, b, c, condition);
}

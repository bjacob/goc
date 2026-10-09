// SPDX-License-Identifier: MIT

#include "rdna4_packed_fma.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_half_fma_scalar.h"
#include "rdna4_packed_alu.h"

#include <stdint.h>

namespace {

int run(uint64_t flags, uint32_t mask, uint32_t mode, uint32_t *const *d, const uint32_t *const *a,
        const uint32_t *const *b, const uint32_t *const *c) {
  if (int error = goc::validate(flags, mode & ~UINT32_C(0x1fff), true))
    return error;
  if (mask == 0)
    return GOC_SUCCESS;
  bool exact = (flags & GOC_SEMANTICS_MASK) == GOC_SEMANTICS_EXACT_EMPIRICAL;
#if defined(GOC_HAVE_X86_64_V3)
  if (!exact && (flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::packed_fma_x86_64_v3(flags & GOC_FP16_OVFL, mask, mode, d[0], a[0], b[0], c[0]);
    return GOC_SUCCESS;
  }
#endif
  goc::HalfFmaEnvironment environment(exact);
  uint32_t modes[] = {goc::packed_half_mode(mode, false), goc::packed_half_mode(mode, true)};
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    result[lane] = 0;
    for (int half = 0; half < 2; ++half) {
      uint32_t m = modes[half];
      uint16_t x = uint16_t(a[0][lane] >> (m & GOC_ALU_HIGH_A ? 16 : 0));
      uint16_t y = uint16_t(b[0][lane] >> (m & GOC_ALU_HIGH_B ? 16 : 0));
      uint16_t z = uint16_t(c[0][lane] >> (m & GOC_ALU_HIGH_C ? 16 : 0));
      result[lane] |= uint32_t(goc::half_fma_value(x, y, z, m, flags & GOC_FP16_OVFL))
                      << (16 * half);
    }
  }
  for (int lane = 0; lane < 32; ++lane)
    if ((mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_pk_fma_f16(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run(flags, mask, mode, d, a, b, c);
}

int goc_rdna4_v_pk_fmac_f16(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b) {
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, mode, true))
    return error;
  return run(flags, mask, 0, d, a, b, d);
}

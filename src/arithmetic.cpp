// SPDX-License-Identifier: MIT

#include "goc_common.h"
#include "goc_rdna4.h"
#include "internal.h"

#include <cmath>
#include <stdint.h>

int goc_rdna4_v_fma_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c) {
  if (int error = goc::validate(flags, instruction_flags))
    return error;

#if defined(GOC_HAVE_AVX512F)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::fma_avx512f(static_cast<uint32_t>(exec_mask), d[0], a[0], b[0], c[0]);
    return GOC_SUCCESS;
  }
#endif

#if defined(GOC_HAVE_AVX2)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::fma_avx2(static_cast<uint32_t>(exec_mask), d[0], a[0], b[0], c[0]);
    return GOC_SUCCESS;
  }
#endif

  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      result[lane] = goc::as_bits(std::fma(goc::as_float(a[0][lane]), goc::as_float(b[0][lane]),
                                           goc::as_float(c[0][lane])));

  for (int lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

int goc_rdna4_v_log_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a) {
  if (int error = goc::validate(flags, instruction_flags))
    return error;

  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      result[lane] = goc::as_bits(std::log2(goc::as_float(a[0][lane])));

  for (int lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

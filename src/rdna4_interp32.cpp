// SPDX-License-Identifier: MIT

// Quad broadcasts follow the RDNA4 ISA guide, sections 12.3 and 16.13.

#include "goc/goc.h"
#include "internal.h"
#include "rdna4_alu.h"
#include "rdna4_interp.h"

#include <cmath>
#include <stdint.h>

namespace {

template <bool P2>
int run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  const uint32_t known =
      GOC_ALU_NEG_A | GOC_ALU_NEG_B | GOC_ALU_NEG_C | GOC_ALU_CLAMP | GOC_INTERP_WAIT_EXP_MASK;
  if (int error = goc::validate(flags, mode & ~known))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::interp32_x86_64_v4<P2>(exec_mask, mode, d[0], a[0], b[0], c[0]);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::interp32_x86_64_v3<P2>(exec_mask, mode, d[0], a[0], b[0], c[0]);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (unsigned lane = 0; lane < 32; ++lane) {
    unsigned quad = lane & ~3u;
    float x = goc::alu_input(a[0][quad + (P2 ? 2 : 1)], mode);
    float y = goc::alu_input(b[0][lane], mode >> 1);
    float z = goc::alu_input(c[0][P2 ? lane : quad], mode >> 2);
    result[lane] = goc::as_bits(goc::alu_output(std::fma(x, y, z), mode));
  }
  for (unsigned lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_interp_p10_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_rdna4_v_interp_p2_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

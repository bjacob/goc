// SPDX-License-Identifier: MIT

#include "rdna4_div_fixup.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_division.h"
#include "rdna4_dpp.h"

#include <stdint.h>

namespace {

template <unsigned Width>
int run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (int error = goc::validate(flags, mode & ~uint32_t(Width == 16 ? 0x1fff : 0x1ff), true))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
  bool saturate = flags & GOC_FP16_OVFL;
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::fixup_x86_64_v4<Width>(exec_mask, mode, saturate, d, a, b, c);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::fixup_x86_64_v3<Width>(exec_mask, mode, saturate, d, a, b, c);
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
    if constexpr (Width == 16) {
      av = (av >> (mode & GOC_ALU_HIGH_A ? 16 : 0)) & 65535;
      bv = (bv >> (mode & GOC_ALU_HIGH_B ? 16 : 0)) & 65535;
      cv = (cv >> (mode & GOC_ALU_HIGH_C ? 16 : 0)) & 65535;
    }
    result[lane] = goc::fixup_value<Width>(av, bv, cv, mode, saturate);
  }
  for (unsigned reg = 0; reg < (Width == 64 ? 2u : 1u); ++reg)
    for (unsigned lane = 0; lane < 32; ++lane)
      if ((exec_mask >> lane) & 1) {
        if constexpr (Width == 16) {
          unsigned shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
          d[0][lane] = (d[0][lane] & ~(65535u << shift)) | (result[lane] << shift);
        } else
          d[reg][lane] = uint32_t(result[lane] >> (Width == 64 ? 32 * reg : 0));
      }
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_div_fixup_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c) {
  if (instruction_flags >> 32)
    return goc::execute_dpp(flags, exec_mask, instruction_flags, a,
                            [&](uint32_t exec_mask, const uint32_t *const *source) {
                              return run<16>(flags, exec_mask, uint32_t(instruction_flags), d,
                                             source, b, c);
                            });
  return run<16>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_rdna4_v_div_fixup_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_rdna4_v_div_fixup_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64>(flags, exec_mask, instruction_flags, d, a, b, c);
}

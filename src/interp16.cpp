// SPDX-License-Identifier: MIT

#include "float_formats.h"
#include "goc/goc.h"
#include "internal.h"
#include "interp.h"
#include "interp16_scalar.h"
#include "mixed_fma_scalar.h"

#include <cmath>
#include <stdint.h>

namespace {

template <bool P2, bool Rtz>
int run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  const uint32_t known = GOC_ALU_NEG_A | GOC_ALU_NEG_B | GOC_ALU_NEG_C | GOC_ALU_CLAMP |
                         GOC_ALU_HIGH_A | (P2 ? GOC_ALU_HIGH_D : GOC_ALU_HIGH_C) |
                         GOC_INTERP_WAIT_EXP_MASK;
  if (int error = goc::validate(flags, mode & ~known))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::interp16_x86_64_v3<P2, Rtz>(flags & GOC_FP16_OVFL, exec_mask, mode, d[0], a[0], b[0],
                                     c[0]);
    return GOC_SUCCESS;
  }
#endif
  unsigned sa = mode & GOC_ALU_HIGH_A ? 16 : 0, sc = mode & GOC_ALU_HIGH_C ? 16 : 0,
           sd = mode & GOC_ALU_HIGH_D ? 16 : 0;
  uint32_t result[32];
  for (unsigned lane = 0; lane < 32; ++lane) {
    unsigned quad = lane & ~3u;
    uint32_t x = goc::as_bits(goc::f16_to_float(uint16_t(a[0][quad + (P2 ? 2 : 1)] >> sa))),
             y = b[0][lane];
    uint32_t z = P2 ? c[0][lane] : goc::as_bits(goc::f16_to_float(uint16_t(c[0][quad] >> sc)));
    if (mode & GOC_ALU_NEG_A)
      x ^= 0x80000000;
    if (mode & GOC_ALU_NEG_B)
      y ^= 0x80000000;
    if (mode & GOC_ALU_NEG_C)
      z ^= 0x80000000;
    if constexpr (P2) {
      uint16_t h =
          Rtz ? goc::interp16_rtz_half(x, y, z, mode & GOC_ALU_CLAMP)
              : goc::mixed_fma_half_value(x, y, z, mode & GOC_ALU_CLAMP, flags & GOC_FP16_OVFL);
      result[lane] = (d[0][lane] & ~(65535u << sd)) | (uint32_t(h) << sd);
    } else {
      float value = Rtz ? goc::interp16_rtz_float(x, y, z)
                        : std::fma(goc::as_float(x), goc::as_float(y), goc::as_float(z));
      if (mode & GOC_ALU_CLAMP)
        value = !(value > 0) ? 0 : value > 1 ? 1 : value;
      result[lane] = goc::as_bits(value);
    }
  }
  for (unsigned lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_v_interp_p10_f16_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<false, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_v_interp_p2_f16_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<true, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_v_interp_p10_rtz_f16_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<false, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_v_interp_p2_rtz_f16_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<true, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

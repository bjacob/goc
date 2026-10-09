// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "internal.h"
#include "rdna4_alu.h"
#include "rdna4_dpp.h"
#include "rdna4_simd.h"

#include <cmath>
#include <stdint.h>

namespace {

template <bool Dx9Zero>
int fma(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (int error = goc::validate(flags, instruction_flags & ~UINT32_C(0x1ff)))
    return error;
  if (uint32_t(exec_mask) == 0)
    return GOC_SUCCESS;

#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    if constexpr (Dx9Zero)
      goc::fma_dx9_zero_x86_64_v4(uint32_t(exec_mask), instruction_flags, d[0], a[0], b[0], c[0]);
    else
      goc::fma_x86_64_v4(static_cast<uint32_t>(exec_mask), instruction_flags, d[0], a[0], b[0],
                         c[0]);
    return GOC_SUCCESS;
  }
#endif

#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    if constexpr (Dx9Zero)
      goc::fma_dx9_zero_x86_64_v3(uint32_t(exec_mask), instruction_flags, d[0], a[0], b[0], c[0]);
    else
      goc::fma_x86_64_v3(static_cast<uint32_t>(exec_mask), instruction_flags, d[0], a[0], b[0],
                         c[0]);
    return GOC_SUCCESS;
  }
#endif

  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    float x = goc::alu_input(a[0][lane], instruction_flags);
    float y = goc::alu_input(b[0][lane], instruction_flags >> 1);
    float z = goc::alu_input(c[0][lane], instruction_flags >> 2);
    float value;
    if constexpr (Dx9Zero)
      value = (x == 0 || y == 0) ? z : std::fma(x, y, z);
    else
      value = std::fma(x, y, z);
    result[lane] = goc::as_bits(goc::alu_output(value, instruction_flags));
  }

  for (int lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

int fma_with_dpp(uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                 const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
                 uint32_t known) {
  const uint64_t dpp_bits = GOC_DPP8 | GOC_DPP_FI | GOC_DPP8_SELECT_MASK;
  if (mode & ~(uint64_t(known) | dpp_bits))
    return GOC_ERROR_INVALID_FLAGS;
  if ((mode >> 32) && !(mode & GOC_DPP8))
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, 0))
    return error;
  if (!uint32_t(mask))
    return GOC_SUCCESS;
  if (mode & GOC_DPP8) {
    uint32_t permuted[32];
    const uint32_t *source = permuted;
    goc::dpp8_source(flags, uint32_t(mask), mode, permuted, a[0]);
    return fma<false>(flags, mask, uint32_t(mode), d, &source, b, c);
  }
  return fma<false>(flags, mask, uint32_t(mode), d, a, b, c);
}

template <bool Multiply>
int literal_fma(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                const uint32_t *const *a, const uint32_t *const *b, uint32_t literal) {
  if (int error = goc::validate(flags, mode))
    return error;
  if (uint32_t(mask) == 0)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::literal_fma_x86_64_v4<Multiply>(uint32_t(mask), literal, d[0], a[0], b[0]);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::literal_fma_x86_64_v3<Multiply>(uint32_t(mask), literal, d[0], a[0], b[0]);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[32];
  float k = goc::as_float(literal);
  for (int lane = 0; lane < 32; ++lane) {
    float x = goc::as_float(a[0][lane]), y = goc::as_float(b[0][lane]);
    result[lane] = goc::as_bits(Multiply ? std::fma(x, k, y) : std::fma(x, y, k));
  }
  for (int lane = 0; lane < 32; ++lane)
    if ((mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_fma_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c) {
  return fma_with_dpp(flags, exec_mask, instruction_flags, d, a, b, c, 0x1ff);
}

int goc_rdna4_v_fma_dx9_zero_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return fma<true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_rdna4_v_fmac_f32(uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b) {
  const uint32_t known = GOC_ALU_NEG_A | GOC_ALU_NEG_B | GOC_ALU_ABS_A | GOC_ALU_ABS_B |
                         GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP;
  return fma_with_dpp(flags, mask, mode, d, a, b, d, known);
}

int goc_rdna4_v_fmamk_f32(uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                          const uint32_t *const *a, uint32_t literal, const uint32_t *const *b) {
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return literal_fma<true>(flags, mask, mode, d, a, b, literal);
}

int goc_rdna4_v_fmaak_f32(uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b, uint32_t literal) {
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return literal_fma<false>(flags, mask, mode, d, a, b, literal);
}

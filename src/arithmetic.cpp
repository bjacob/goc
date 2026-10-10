// SPDX-License-Identifier: MIT
// DX9 flushing and zero-product addition follow rocjitsu fp_mode::arithmetic.

#include "alu.h"
#include "dpp.h"
#include "goc/goc.h"
#include "internal.h"
#include "simd.h"

#include <cmath>
#include <stdint.h>

namespace {

template <bool Dx9Zero, bool Rdna3 = false>
int fma(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if constexpr (Rdna3)
    if (!(flags & GOC_FP_FLUSH_OUTPUT_DENORMALS) || (flags & GOC_FP_IEEE_MODE))
      instruction_flags &= ~GOC_ALU_OMOD_HALF;

#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    if constexpr (Dx9Zero)
      goc::fma_dx9_zero_x86_64_v4(exec_mask, instruction_flags, d[0], a[0], b[0], c[0]);
    else
      goc::fma_x86_64_v4(flags, exec_mask, instruction_flags, d[0], a[0], b[0], c[0]);
    return GOC_SUCCESS;
  }
#endif

#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    if constexpr (Dx9Zero)
      goc::fma_dx9_zero_x86_64_v3(exec_mask, instruction_flags, d[0], a[0], b[0], c[0]);
    else
      goc::fma_x86_64_v3(flags, exec_mask, instruction_flags, d[0], a[0], b[0], c[0]);
    return GOC_SUCCESS;
  }
#endif

  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    float x = goc::alu_input(a[0][lane], instruction_flags);
    float y = goc::alu_input(b[0][lane], instruction_flags >> 1);
    float z = goc::alu_input(c[0][lane], instruction_flags >> 2);
    if (Dx9Zero || (flags & GOC_FP_FLUSH_INPUT_DENORMALS)) {
      x = goc::flush_denorm_f32(x);
      y = goc::flush_denorm_f32(y);
      z = goc::flush_denorm_f32(z);
    }
    if constexpr (Dx9Zero) {
      if (x == 0 || y == 0) {
        x = 0;
        y = 1;
      }
    }
    float value = std::fma(x, y, z);
    if (Dx9Zero || (flags & GOC_FP_FLUSH_OUTPUT_DENORMALS))
      value = goc::flush_denorm_f32(value);
    result[lane] = goc::as_bits(goc::alu_output_f32(value, instruction_flags));
  }

  for (int lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

// Validates DPP routing and stages A before any destination writes.
template <bool Rdna3, bool Dx9Zero>
[[gnu::noinline]] int fma_dpp(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c) {
  if (!goc::valid_dpp(mode))
    return GOC_ERROR_INVALID_FLAGS;
  if (!exec_mask)
    return GOC_SUCCESS;
  uint32_t permuted[32];
  const uint32_t *source = permuted;
  if (mode & GOC_DPP8)
    goc::dpp8_source(flags, exec_mask, mode, permuted, a[0]);
  else
    exec_mask = goc::dpp16_source(flags, exec_mask, mode, permuted, a[0]);
  if (!exec_mask)
    return GOC_SUCCESS;
  return fma<Dx9Zero, Rdna3>(flags, exec_mask, uint32_t(mode), d, &source, b, c);
}

template <bool Rdna3 = false, bool Dx9Zero = false>
int fma_with_dpp(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                 const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
                 uint32_t known) {
  if (uint32_t(mode) & ~known)
    return GOC_ERROR_INVALID_FLAGS;
  constexpr uint64_t fp_flags =
      GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS | GOC_FP_IEEE_MODE;
  if (int error = goc::validate(flags, 0, false, fp_flags))
    return error;
  if (mode >> 32)
    return fma_dpp<Rdna3, Dx9Zero>(flags, exec_mask, mode, d, a, b, c);
  if (!exec_mask)
    return GOC_SUCCESS;
  return fma<Dx9Zero, Rdna3>(flags, exec_mask, uint32_t(mode), d, a, b, c);
}

template <bool Multiply>
int literal_fma(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                const uint32_t *const *a, const uint32_t *const *b, uint32_t literal) {
  if (int error = goc::validate(flags, mode, false,
                                GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS |
                                    GOC_FP_IEEE_MODE))
    return error;
  if (exec_mask == 0)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::literal_fma_x86_64_v4<Multiply>(flags, exec_mask, literal, d[0], a[0], b[0]);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::literal_fma_x86_64_v3<Multiply>(flags, exec_mask, literal, d[0], a[0], b[0]);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[32];
  float k = goc::as_float(literal);
  if (flags & GOC_FP_FLUSH_INPUT_DENORMALS)
    k = goc::flush_denorm_f32(k);
  for (int lane = 0; lane < 32; ++lane) {
    float x = goc::as_float(a[0][lane]), y = goc::as_float(b[0][lane]);
    if (flags & GOC_FP_FLUSH_INPUT_DENORMALS) {
      x = goc::flush_denorm_f32(x);
      y = goc::flush_denorm_f32(y);
    }
    float value = Multiply ? std::fma(x, k, y) : std::fma(x, y, k);
    if (flags & GOC_FP_FLUSH_OUTPUT_DENORMALS)
      value = goc::flush_denorm_f32(value);
    result[lane] = goc::as_bits(value);
  }
  for (int lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

template <bool Dx9Zero, bool Rdna3 = false>
int fma_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  int error =
      fma_with_dpp<Rdna3, Dx9Zero>(flags, 0, mode, nullptr, nullptr, nullptr, nullptr, 0x1ff);
  if (error || !exec_mask)
    return error;
  for (unsigned half = 0; half < 2; ++half) {
    uint64_t half_mode = mode;
    if (mode & GOC_DPP16) {
      uint64_t rows = (mode & GOC_DPP_ROW_MASK) >> GOC_DPP_ROW_SHIFT;
      half_mode = (mode & ~GOC_DPP_ROW_MASK) | (((rows >> (half * 2)) & 3) << GOC_DPP_ROW_SHIFT);
    }
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *x = a[0] + half * 32, *y = b[0] + half * 32, *z = c[0] + half * 32;
    uint32_t half_exec_mask = uint32_t(exec_mask >> (half * 32));
    error =
        fma_with_dpp<Rdna3, Dx9Zero>(flags, half_exec_mask, half_mode, &dest, &x, &y, &z, 0x1ff);
    if (error)
      return error;
  }
  return GOC_SUCCESS;
}

template <bool Multiply>
int literal_fma_wave64(uint64_t flags, uint64_t exec_mask, uint32_t mode, uint32_t *const *d,
                       const uint32_t *const *a, const uint32_t *const *b, uint32_t literal) {
  if (int error = literal_fma<Multiply>(flags, 0, mode, nullptr, nullptr, nullptr, literal))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
  for (unsigned half = 0; half < 2; ++half) {
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *x = a[0] + half * 32, *y = b[0] + half * 32;
    if (int error = literal_fma<Multiply>(flags, uint32_t(exec_mask >> (half * 32)), mode, &dest,
                                          &x, &y, literal))
      return error;
  }
  return GOC_SUCCESS;
}

} // namespace

int goc_v_fma_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return fma_with_dpp(flags, exec_mask, instruction_flags, d, a, b, c, 0x1ff);
}

int goc_v_fma_dx9_zero_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c) {
  return fma_with_dpp<false, true>(flags, exec_mask, instruction_flags, d, a, b, c, 0x1ff);
}

int goc_v_fmac_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b,
                         uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  const uint32_t known = GOC_ALU_NEG_A | GOC_ALU_NEG_B | GOC_ALU_ABS_A | GOC_ALU_ABS_B |
                         GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP;
  return fma_with_dpp(flags, exec_mask, mode, d, a, b, d, known);
}

int goc_v_fmamk_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                          const uint32_t *const *a, uint32_t literal, const uint32_t *const *b,
                          uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return literal_fma<true>(flags, exec_mask, mode, d, a, b, literal);
}

int goc_v_fmaak_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b, uint32_t literal,
                          uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return literal_fma<false>(flags, exec_mask, mode, d, a, b, literal);
}

int goc_v_fma_f32_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c,
                               uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return fma_wave64<false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_v_fma_dx9_zero_f32_rdna4_wave64(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c) {
  return fma_wave64<true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_v_fmac_f32_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  const uint32_t known = GOC_ALU_NEG_A | GOC_ALU_NEG_B | GOC_ALU_ABS_A | GOC_ALU_ABS_B |
                         GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP;
  if (uint32_t(mode) & ~known)
    return GOC_ERROR_INVALID_FLAGS;
  return fma_wave64<false>(flags, exec_mask, mode, d, a, b, d);
}

int goc_v_fmamk_f32_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                 uint32_t *const *d, const uint32_t *const *a, uint32_t literal,
                                 const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return literal_fma_wave64<true>(flags, exec_mask, mode, d, a, b, literal);
}

int goc_v_fmaak_f32_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, uint32_t literal,
                                 uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return literal_fma_wave64<false>(flags, exec_mask, mode, d, a, b, literal);
}

int goc_v_fma_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                  uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                  const uint32_t *const *c, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return fma_with_dpp<true>(flags, exec_mask, instruction_flags, d, a, b, c, 0x1ff);
}

int goc_v_fma_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return fma_wave64<false, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_v_fmac_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  const uint32_t known = GOC_ALU_NEG_A | GOC_ALU_NEG_B | GOC_ALU_ABS_A | GOC_ALU_ABS_B |
                         GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP;
  return fma_with_dpp<true>(flags, exec_mask, mode, d, a, b, d, known);
}

int goc_v_fmac_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  const uint32_t known = GOC_ALU_NEG_A | GOC_ALU_NEG_B | GOC_ALU_ABS_A | GOC_ALU_ABS_B |
                         GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP;
  if (uint32_t(mode) & ~known)
    return GOC_ERROR_INVALID_FLAGS;
  return fma_wave64<false, true>(flags, exec_mask, mode, d, a, b, d);
}

int goc_v_fma_dx9_zero_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  return fma_with_dpp<true, true>(flags, exec_mask, instruction_flags, d, a, b, c, 0x1ff);
}

int goc_v_fma_dx9_zero_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c) {
  return fma_wave64<true, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_v_fmac_dx9_zero_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a,
                            const uint32_t *const *b) {
  const uint32_t known = GOC_ALU_NEG_A | GOC_ALU_NEG_B | GOC_ALU_ABS_A | GOC_ALU_ABS_B |
                         GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP;
  if (uint32_t(instruction_flags) & ~known)
    return GOC_ERROR_INVALID_FLAGS;
  return goc_v_fma_dx9_zero_f32(flags, exec_mask, instruction_flags, d, a, b, d);
}

int goc_v_fmac_dx9_zero_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b) {
  const uint32_t known = GOC_ALU_NEG_A | GOC_ALU_NEG_B | GOC_ALU_ABS_A | GOC_ALU_ABS_B |
                         GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP;
  if (uint32_t(instruction_flags) & ~known)
    return GOC_ERROR_INVALID_FLAGS;
  return goc_v_fma_dx9_zero_f32_wave64(flags, exec_mask, instruction_flags, d, a, b, d);
}

int goc_v_fmamk_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                    const uint32_t *const *a, uint32_t literal, const uint32_t *const *b,
                    uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return literal_fma<true>(flags, exec_mask, mode, d, a, b, literal);
}

int goc_v_fmaak_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                    const uint32_t *const *a, const uint32_t *const *b, uint32_t literal,
                    uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return literal_fma<false>(flags, exec_mask, mode, d, a, b, literal);
}

int goc_v_fmamk_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, uint32_t literal, const uint32_t *const *b,
                           uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return literal_fma_wave64<true>(flags, exec_mask, mode, d, a, b, literal);
}

int goc_v_fmaak_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b, uint32_t literal,
                           uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return literal_fma_wave64<false>(flags, exec_mask, mode, d, a, b, literal);
}

// SPDX-License-Identifier: MIT

#include "half_fma.h"
#include "dpp.h"
#include "fma.h"
#include "goc/goc.h"
#include "half_fma_scalar.h"
#include "internal.h"

#include <stdint.h>

namespace {

static const uint64_t fp_flags =
    GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS | GOC_FP_IEEE_MODE;

template <goc::FmaOperands Operands = goc::FmaOperands::Registers, bool Rdna3 = false>
int run(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
        uint32_t *excp_flag_user, uint16_t literal = 0) {
  bool report = excp_flag_user && (flags & GOC_SEMANTICS_MASK) == GOC_SEMANTICS_EXACT_EMPIRICAL;
  if (mode >> 32) {
    if constexpr (Operands == goc::FmaOperands::Registers) {
      return goc::execute_dpp(flags, exec_mask, mode, a,
                              [&](uint32_t exec_mask, const uint32_t *const *source) {
                                return run<Operands, Rdna3>(flags, exec_mask, uint32_t(mode), d,
                                                            source, b, c, excp_flag_user, literal);
                              });
    } else {
      return GOC_ERROR_INVALID_FLAGS;
    }
  }
  const uint32_t known = Operands == goc::FmaOperands::Registers
                             ? 0x1fffU
                             : GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_D;
  if (int error = goc::validate(flags, mode & ~known, true, fp_flags))
    return error;
  if constexpr (Rdna3)
    if (!(flags & GOC_FP_FLUSH_OUTPUT_DENORMALS) || (flags & GOC_FP_IEEE_MODE))
      mode &= ~uint64_t(GOC_ALU_OMOD_HALF);
  if (exec_mask == 0)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V3)
  bool exact = (flags & GOC_SEMANTICS_MASK) == GOC_SEMANTICS_EXACT_EMPIRICAL;
  if (!exact && (flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::half_fma_x86_64_v3<Operands>(flags, exec_mask, mode, d[0], a[0], b[0],
                                      Operands == goc::FmaOperands::Registers ? c[0] : nullptr,
                                      literal);
    return GOC_SUCCESS;
  }
#endif
  int a_shift = mode & GOC_ALU_HIGH_A ? 16 : 0;
  int b_shift = mode & GOC_ALU_HIGH_B ? 16 : 0;
  int c_shift = mode & GOC_ALU_HIGH_C ? 16 : 0;
  int d_shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  uint32_t exceptions = 0;
  uint16_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    uint16_t x = uint16_t(a[0][lane] >> a_shift), y = uint16_t(b[0][lane] >> b_shift), z;
    if constexpr (Operands == goc::FmaOperands::MultiplyLiteral) {
      z = y;
      y = literal;
    } else if constexpr (Operands == goc::FmaOperands::AddLiteral) {
      z = literal;
    } else {
      z = uint16_t(c[0][lane] >> c_shift);
    }
    result[lane] = goc::half_fma_value<Rdna3>(
        x, y, z, mode, flags, report && ((exec_mask >> lane) & 1) ? &exceptions : nullptr);
  }
  for (int lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = (d[0][lane] & ~(0xffffU << d_shift)) | (uint32_t(result[lane]) << d_shift);
  if (report)
    *excp_flag_user |= exceptions;
  return GOC_SUCCESS;
}

template <goc::FmaOperands Operands = goc::FmaOperands::Registers, bool Rdna3 = false>
int run_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
               uint32_t *excp_flag_user, uint16_t literal = 0) {
  if (int error = run<Operands, Rdna3>(flags, 0, mode, nullptr, nullptr, nullptr, nullptr,
                                       excp_flag_user, literal))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
  bool report = excp_flag_user && (flags & GOC_SEMANTICS_MASK) == GOC_SEMANTICS_EXACT_EMPIRICAL;
  uint32_t exceptions = 0;
  for (unsigned half = 0; half < 2; ++half) {
    uint64_t half_mode = mode;
    if (mode & GOC_DPP16) {
      uint64_t rows = (mode & GOC_DPP_ROW_MASK) >> GOC_DPP_ROW_SHIFT;
      half_mode = (mode & ~GOC_DPP_ROW_MASK) | (((rows >> (half * 2)) & 3) << GOC_DPP_ROW_SHIFT);
    }
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *x = a[0] + half * 32, *y = b[0] + half * 32;
    const uint32_t *z = Operands == goc::FmaOperands::Registers ? c[0] + half * 32 : nullptr;
    if (int error =
            run<Operands, Rdna3>(flags, uint32_t(exec_mask >> (half * 32)), half_mode, &dest, &x,
                                 &y, &z, report ? &exceptions : nullptr, literal))
      return error;
  }
  // The global output may alias an upper-half source: update only after all reads.
  if (report)
    *excp_flag_user |= exceptions;
  return GOC_SUCCESS;
}

} // namespace

int goc_v_fma_f16_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c, uint32_t *excp_flag_user) {
  if (excp_flag_user && goc::half_fma_reporting_unsupported(flags, false))
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run(flags, exec_mask, mode, d, a, b, c, excp_flag_user);
}

int goc_v_fmac_f16_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b,
                         uint32_t *excp_flag_user) {
  if (excp_flag_user && goc::half_fma_reporting_unsupported(flags, false))
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return goc::execute_dpp(flags, exec_mask, mode, a,
                            [&](uint32_t exec_mask, const uint32_t *const *source) {
                              return goc_v_fmac_f16_rdna4(flags, exec_mask, uint32_t(mode), d,
                                                          source, b, excp_flag_user);
                            });
  const uint32_t known = GOC_ALU_NEG_A | GOC_ALU_NEG_B | GOC_ALU_ABS_A | GOC_ALU_ABS_B |
                         GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP | GOC_ALU_HIGH_A | GOC_ALU_HIGH_B |
                         GOC_ALU_HIGH_D;
  if (int error = goc::validate(flags, mode & ~known, true, fp_flags))
    return error;
  // FMAC reads the same destination half that it overwrites.
  if (mode & GOC_ALU_HIGH_D)
    mode |= GOC_ALU_HIGH_C;
  return run(flags, exec_mask, mode, d, a, b, d, excp_flag_user);
}

int goc_v_fmamk_f16_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                          const uint32_t *const *a, uint16_t literal, const uint32_t *const *b,
                          uint32_t *excp_flag_user) {
  if (excp_flag_user && goc::half_fma_reporting_unsupported(flags, false))
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<goc::FmaOperands::MultiplyLiteral>(flags, exec_mask, mode, d, a, b, nullptr,
                                                excp_flag_user, literal);
}

int goc_v_fmaak_f16_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b, uint16_t literal,
                          uint32_t *excp_flag_user) {
  if (excp_flag_user && goc::half_fma_reporting_unsupported(flags, false))
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<goc::FmaOperands::AddLiteral>(flags, exec_mask, mode, d, a, b, nullptr, excp_flag_user,
                                           literal);
}

int goc_v_fma_f16_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c,
                               uint32_t *excp_flag_user) {
  if (excp_flag_user && goc::half_fma_reporting_unsupported(flags, false))
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run_wave64(flags, exec_mask, mode, d, a, b, c, excp_flag_user);
}

int goc_v_fmac_f16_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && goc::half_fma_reporting_unsupported(flags, false))
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if ((mode >> 32) && !goc::valid_dpp(mode))
    return GOC_ERROR_INVALID_FLAGS;
  const uint32_t known = GOC_ALU_NEG_A | GOC_ALU_NEG_B | GOC_ALU_ABS_A | GOC_ALU_ABS_B |
                         GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP | GOC_ALU_HIGH_A | GOC_ALU_HIGH_B |
                         GOC_ALU_HIGH_D;
  if (int error = goc::validate(flags, uint32_t(mode) & ~known, true, fp_flags))
    return error;
  // FMAC reads the same destination half that it overwrites.
  if (mode & GOC_ALU_HIGH_D)
    mode |= GOC_ALU_HIGH_C;
  return run_wave64(flags, exec_mask, mode, d, a, b, d, excp_flag_user);
}

int goc_v_fmamk_f16_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                 uint32_t *const *d, const uint32_t *const *a, uint16_t literal,
                                 const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && goc::half_fma_reporting_unsupported(flags, false))
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run_wave64<goc::FmaOperands::MultiplyLiteral>(flags, exec_mask, mode, d, a, b, nullptr,
                                                       excp_flag_user, literal);
}

int goc_v_fmaak_f16_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, uint16_t literal,
                                 uint32_t *excp_flag_user) {
  if (excp_flag_user && goc::half_fma_reporting_unsupported(flags, false))
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run_wave64<goc::FmaOperands::AddLiteral>(flags, exec_mask, mode, d, a, b, nullptr,
                                                  excp_flag_user, literal);
}

int goc_v_fma_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
                  uint32_t *trapsts) {
  if (trapsts && goc::half_fma_reporting_unsupported(flags, true))
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<goc::FmaOperands::Registers, true>(flags, exec_mask, mode, d, a, b, c, trapsts);
}

int goc_v_fma_f16_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c, uint32_t *trapsts) {
  if (trapsts && goc::half_fma_reporting_unsupported(flags, true))
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run_wave64<goc::FmaOperands::Registers, true>(flags, exec_mask, mode, d, a, b, c, trapsts);
}

int goc_v_fmac_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts) {
  if (trapsts && goc::half_fma_reporting_unsupported(flags, true))
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return goc::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return goc_v_fmac_f16(flags, exec_mask, uint32_t(mode), d, source, b, trapsts);
        });
  const uint32_t known = GOC_ALU_NEG_A | GOC_ALU_NEG_B | GOC_ALU_ABS_A | GOC_ALU_ABS_B |
                         GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP | GOC_ALU_HIGH_A | GOC_ALU_HIGH_B |
                         GOC_ALU_HIGH_D;
  if (int error = goc::validate(flags, mode & ~known, true, fp_flags))
    return error;
  // FMAC reads the same destination half that it overwrites.
  if (mode & GOC_ALU_HIGH_D)
    mode |= GOC_ALU_HIGH_C;
  return run<goc::FmaOperands::Registers, true>(flags, exec_mask, mode, d, a, b, d, trapsts);
}

int goc_v_fmac_f16_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts) {
  if (trapsts && goc::half_fma_reporting_unsupported(flags, true))
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if ((mode >> 32) && !goc::valid_dpp(mode))
    return GOC_ERROR_INVALID_FLAGS;
  const uint32_t known = GOC_ALU_NEG_A | GOC_ALU_NEG_B | GOC_ALU_ABS_A | GOC_ALU_ABS_B |
                         GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP | GOC_ALU_HIGH_A | GOC_ALU_HIGH_B |
                         GOC_ALU_HIGH_D;
  if (int error = goc::validate(flags, uint32_t(mode) & ~known, true, fp_flags))
    return error;
  // FMAC reads the same destination half that it overwrites.
  if (mode & GOC_ALU_HIGH_D)
    mode |= GOC_ALU_HIGH_C;
  return run_wave64<goc::FmaOperands::Registers, true>(flags, exec_mask, mode, d, a, b, d, trapsts);
}

int goc_v_fmamk_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                    const uint32_t *const *a, uint16_t literal, const uint32_t *const *b,
                    uint32_t *trapsts) {
  if (trapsts && goc::half_fma_reporting_unsupported(flags, true))
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<goc::FmaOperands::MultiplyLiteral, true>(flags, exec_mask, mode, d, a, b, nullptr,
                                                      trapsts, literal);
}

int goc_v_fmamk_f16_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, uint16_t literal, const uint32_t *const *b,
                           uint32_t *trapsts) {
  if (trapsts && goc::half_fma_reporting_unsupported(flags, true))
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run_wave64<goc::FmaOperands::MultiplyLiteral, true>(flags, exec_mask, mode, d, a, b,
                                                             nullptr, trapsts, literal);
}

int goc_v_fmaak_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                    const uint32_t *const *a, const uint32_t *const *b, uint16_t literal,
                    uint32_t *trapsts) {
  if (trapsts && goc::half_fma_reporting_unsupported(flags, true))
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<goc::FmaOperands::AddLiteral, true>(flags, exec_mask, mode, d, a, b, nullptr, trapsts,
                                                 literal);
}

int goc_v_fmaak_f16_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b, uint16_t literal,
                           uint32_t *trapsts) {
  if (trapsts && goc::half_fma_reporting_unsupported(flags, true))
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run_wave64<goc::FmaOperands::AddLiteral, true>(flags, exec_mask, mode, d, a, b, nullptr,
                                                        trapsts, literal);
}

// SPDX-License-Identifier: MIT

#include "div_fixup.h"
#include "division.h"
#include "dpp.h"
#include "goc/goc.h"
#include "internal.h"

#include <stdint.h>

namespace {

template <unsigned Width, bool Rdna3Half = false>
int run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
        uint32_t *excp_flag_user);

// Flush only denominator/numerator subnormals, retaining their signs and all
// provisional-quotient bits. All source reads precede the instruction's stores.
template <unsigned Width, bool Rdna3Half>
[[gnu::noinline]] int run_flushed(uint64_t flags, uint32_t exec_mask, uint32_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c,
                                  uint32_t *excp_flag_user) {
  using F = goc::DivisionFormat<Width>;
  using T = typename F::Bits;
  uint32_t words[2][(Width == 64 ? 2 : 1)][32];
  const uint32_t *sources[2][(Width == 64 ? 2 : 1)];
  for (unsigned operand = 0; operand < 2; ++operand) {
    const uint32_t *const *input = operand ? c : b;
    for (unsigned lane = 0; lane < 32; ++lane) {
      T value = input[0][lane];
      if constexpr (Width == 64)
        value |= T(input[1][lane]) << 32;
      unsigned shift = 0;
      if constexpr (Width == 16) {
        shift = mode & (GOC_ALU_HIGH_B << operand) ? 16 : 0;
        value = (value >> shift) & 65535;
      }
      if (!(value & F::infinity))
        value &= F::sign;
      if constexpr (Width == 16)
        value <<= shift;
      for (unsigned reg = 0; reg < (Width == 64 ? 2 : 1); ++reg)
        words[operand][reg][lane] = uint32_t(value >> (reg * 32));
    }
    for (unsigned reg = 0; reg < (Width == 64 ? 2 : 1); ++reg)
      sources[operand][reg] = words[operand][reg];
  }
  return run<Width, Rdna3Half>(flags & ~GOC_FP_FLUSH_INPUT_DENORMALS, exec_mask, mode, d, a,
                               sources[0], sources[1], excp_flag_user);
}

template <unsigned Width, bool Rdna3Half>
int run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
        uint32_t *excp_flag_user) {
  constexpr uint64_t fp_flags =
      GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS | GOC_FP_IEEE_MODE;
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE &&
      (flags & GOC_FP_FLUSH_INPUT_DENORMALS))
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (int error =
          goc::validate(flags, mode & ~uint32_t(Width == 16 ? 0x1fff : 0x1ff), true, fp_flags))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
  if (flags & GOC_FP_FLUSH_INPUT_DENORMALS)
    return run_flushed<Width, Rdna3Half>(flags, exec_mask, mode, d, a, b, c, excp_flag_user);
  bool report = excp_flag_user && (flags & GOC_SEMANTICS_MASK) == GOC_SEMANTICS_EXACT_EMPIRICAL;
  uint32_t exceptions = 0;
  bool saturate = flags & GOC_FP16_OVFL;
#if defined(GOC_HAVE_X86_64_V4)
  if (!report && (flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::fixup_x86_64_v4<Width, Rdna3Half>(exec_mask, mode, saturate, d, a, b, c);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if (!report && (flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::fixup_x86_64_v3<Width, Rdna3Half>(exec_mask, mode, saturate, d, a, b, c);
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
    result[lane] = goc::fixup_value<Width, Rdna3Half>(av, bv, cv, mode, saturate);
    if (report && ((exec_mask >> lane) & 1))
      exceptions |= goc::fixup_exceptions<Width>(av, bv, cv, mode);
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
  if (report)
    *excp_flag_user |= exceptions;
  return GOC_SUCCESS;
}

template <unsigned Width, bool Rdna3 = false>
int run_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
               uint32_t *excp_flag_user) {
  constexpr auto instruction =
      Width == 16   ? (Rdna3 ? goc_v_div_fixup_f16 : goc_v_div_fixup_f16_rdna4)
      : Width == 32 ? (Rdna3 ? goc_v_div_fixup_f32 : goc_v_div_fixup_f32_rdna4)
                    : (Rdna3 ? goc_v_div_fixup_f64 : goc_v_div_fixup_f64_rdna4);
  if (int error = instruction(flags, 0, mode, nullptr, nullptr, nullptr, nullptr, excp_flag_user))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
  bool report = excp_flag_user && (flags & GOC_SEMANTICS_MASK) == GOC_SEMANTICS_EXACT_EMPIRICAL;
  // Delay the global-register write until all potentially aliased inputs are read.
  uint32_t exceptions = 0;
  for (unsigned half = 0; half < 2; ++half) {
    uint64_t half_mode = mode;
    if (mode & GOC_DPP16) {
      uint64_t rows = (mode & GOC_DPP_ROW_MASK) >> GOC_DPP_ROW_SHIFT;
      half_mode = (mode & ~GOC_DPP_ROW_MASK) | (((rows >> (half * 2)) & 3) << GOC_DPP_ROW_SHIFT);
    }
    uint32_t *dest[Width == 64 ? 2 : 1];
    const uint32_t *av[Width == 64 ? 2 : 1], *bv[Width == 64 ? 2 : 1], *cv[Width == 64 ? 2 : 1];
    for (unsigned reg = 0; reg < (Width == 64 ? 2U : 1U); ++reg) {
      dest[reg] = d[reg] + half * 32;
      av[reg] = a[reg] + half * 32;
      bv[reg] = b[reg] + half * 32;
      cv[reg] = c[reg] + half * 32;
    }
    if (int error = instruction(flags, uint32_t(exec_mask >> (half * 32)), half_mode, dest, av, bv,
                                cv, report ? &exceptions : nullptr))
      return error;
  }
  if (report)
    *excp_flag_user |= exceptions;
  return GOC_SUCCESS;
}

} // namespace

int goc_v_div_fixup_f16_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c,
                              uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return goc::execute_dpp(flags, exec_mask, instruction_flags, a,
                            [&](uint32_t exec_mask, const uint32_t *const *source) {
                              return run<16>(flags, exec_mask, uint32_t(instruction_flags), d,
                                             source, b, c, excp_flag_user);
                            });
  return run<16>(flags, exec_mask, instruction_flags, d, a, b, c, excp_flag_user);
}

int goc_v_div_fixup_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c,
                              uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32>(flags, exec_mask, instruction_flags, d, a, b, c, excp_flag_user);
}

int goc_v_div_fixup_f64_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c,
                              uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64>(flags, exec_mask, instruction_flags, d, a, b, c, excp_flag_user);
}

int goc_v_div_fixup_f16_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, const uint32_t *const *c,
                                     uint32_t *excp_flag_user) {
  return run_wave64<16>(flags, exec_mask, instruction_flags, d, a, b, c, excp_flag_user);
}

int goc_v_div_fixup_f32_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, const uint32_t *const *c,
                                     uint32_t *excp_flag_user) {
  return run_wave64<32>(flags, exec_mask, instruction_flags, d, a, b, c, excp_flag_user);
}

int goc_v_div_fixup_f64_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, const uint32_t *const *c,
                                     uint32_t *excp_flag_user) {
  return run_wave64<64>(flags, exec_mask, instruction_flags, d, a, b, c, excp_flag_user);
}

int goc_v_div_fixup_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (!(flags & GOC_FP_FLUSH_OUTPUT_DENORMALS) || (flags & GOC_FP_IEEE_MODE))
    instruction_flags &= ~uint64_t(GOC_ALU_OMOD_HALF);
  return run<32>(flags, exec_mask, instruction_flags, d, a, b, c, nullptr);
}

int goc_v_div_fixup_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c,
                               uint32_t *trapsts) {
  return run_wave64<32, true>(flags, exec_mask, instruction_flags, d, a, b, c, trapsts);
}

int goc_v_div_fixup_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (!(flags & GOC_FP_FLUSH_OUTPUT_DENORMALS) || (flags & GOC_FP_IEEE_MODE))
    instruction_flags &= ~uint64_t(GOC_ALU_OMOD_HALF);
  return run<64>(flags, exec_mask, instruction_flags, d, a, b, c, nullptr);
}

int goc_v_div_fixup_f64_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c,
                               uint32_t *trapsts) {
  return run_wave64<64, true>(flags, exec_mask, instruction_flags, d, a, b, c, trapsts);
}

int goc_v_div_fixup_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (!(flags & GOC_FP_FLUSH_OUTPUT_DENORMALS) || (flags & GOC_FP_IEEE_MODE))
    instruction_flags &= ~uint64_t(GOC_ALU_OMOD_HALF);
  if (instruction_flags >> 32)
    return goc::execute_dpp(flags, exec_mask, instruction_flags, a,
                            [&](uint32_t exec_mask, const uint32_t *const *source) {
                              return run<16, true>(flags, exec_mask, uint32_t(instruction_flags), d,
                                                   source, b, c, nullptr);
                            });
  return run<16, true>(flags, exec_mask, instruction_flags, d, a, b, c, nullptr);
}

int goc_v_div_fixup_f16_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c,
                               uint32_t *trapsts) {
  return run_wave64<16, true>(flags, exec_mask, instruction_flags, d, a, b, c, trapsts);
}

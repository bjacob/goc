// SPDX-License-Identifier: MIT

#include "div_fmas.h"
#include "division.h"
#include "goc/goc.h"
#include "internal.h"

#include <stdint.h>

namespace {

// Evaluate all source lanes before masked stores, including overlapping VGPRs.
template <unsigned Width>
[[gnu::noinline]] void scalar(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c, uint32_t condition, uint32_t *exceptions,
                              bool flush_output) {
  using T = typename goc::DivisionFormat<Width>::Bits;
  T result[32];
  for (unsigned lane = 0; lane < 32; ++lane) {
    T av = a[0][lane], bv = b[0][lane], cv = c[0][lane];
    if constexpr (Width == 64) {
      av |= T(a[1][lane]) << 32;
      bv |= T(b[1][lane]) << 32;
      cv |= T(c[1][lane]) << 32;
    }
    result[lane] = goc::division_fmas<Width>(av, bv, cv, (condition >> lane) & 1, mode,
                                             ((exec_mask >> lane) & 1) ? exceptions : nullptr);
  }
  if (flush_output)
    for (auto &value : result)
      if (!(value & goc::DivisionFormat<Width>::infinity))
        value &= goc::DivisionFormat<Width>::sign;
  for (unsigned reg = 0; reg < (Width == 64 ? 2u : 1u); ++reg)
    for (unsigned lane = 0; lane < 32; ++lane)
      if ((exec_mask >> lane) & 1)
        d[reg][lane] = uint32_t(result[lane] >> (Width == 64 ? 32 * reg : 0));
}

template <unsigned Width, bool Rdna3>
int run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
        uint32_t condition, uint32_t *global_state) {
  // RDNA3 output scaling is disabled with IEEE or preserved output denormals.
  if constexpr (Rdna3)
    if (!(flags & GOC_FP_FLUSH_OUTPUT_DENORMALS) || (flags & GOC_FP_IEEE_MODE))
      mode &= ~GOC_ALU_OMOD_HALF;
  constexpr uint64_t fp_flags =
      GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS | GOC_FP_IEEE_MODE;
  if (global_state && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE &&
      (Rdna3 || (flags & GOC_FP_FLUSH_OUTPUT_DENORMALS) ||
       (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_EXACT_EMPIRICAL))
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  // The upstream integer model supplies exact unscaled results on both GPUs.
  // RDNA3 active OMOD has no borrowed exact model; retain the loose path.
  if (int error =
          goc::validate(flags, mode & ~511u, !Rdna3 || !(mode & GOC_ALU_OMOD_HALF), fp_flags))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
  bool report = global_state && (flags & GOC_SEMANTICS_MASK) == GOC_SEMANTICS_EXACT_EMPIRICAL;
  uint32_t exceptions = 0;
  bool flush_output = flags & GOC_FP_FLUSH_OUTPUT_DENORMALS;
#if defined(GOC_HAVE_X86_64_V4)
  if (!report && (flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::div_fmas_x86_64_v4<Width>(exec_mask, mode, d, a, b, c, condition, flush_output);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if (!report && (flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::div_fmas_x86_64_v3<Width>(exec_mask, mode, d, a, b, c, condition, flush_output);
    return GOC_SUCCESS;
  }
#endif
  scalar<Width>(exec_mask, mode, d, a, b, c, condition, report ? &exceptions : nullptr,
                flush_output);
  if (report)
    *global_state |= exceptions;
  return GOC_SUCCESS;
}

template <unsigned Width, bool Rdna3>
int run_wave64(uint64_t flags, uint64_t exec_mask, uint32_t mode, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
               uint64_t condition, uint32_t *global_state) {
  if (int error =
          run<Width, Rdna3>(flags, 0, mode, nullptr, nullptr, nullptr, nullptr, 0, global_state))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
  bool report = global_state && (flags & GOC_SEMANTICS_MASK) == GOC_SEMANTICS_EXACT_EMPIRICAL;
  // Consume both halves before updating a potentially aliased global register.
  uint32_t exceptions = 0;
  for (unsigned half = 0; half < 2; ++half) {
    uint32_t *dest[Width / 32];
    const uint32_t *av[Width / 32], *bv[Width / 32], *cv[Width / 32];
    for (unsigned reg = 0; reg < Width / 32; ++reg) {
      dest[reg] = d[reg] + half * 32;
      av[reg] = a[reg] + half * 32;
      bv[reg] = b[reg] + half * 32;
      cv[reg] = c[reg] + half * 32;
    }
    if (int error =
            run<Width, Rdna3>(flags, uint32_t(exec_mask >> (half * 32)), mode, dest, av, bv, cv,
                              uint32_t(condition >> (half * 32)), report ? &exceptions : nullptr))
      return error;
  }
  if (report)
    *global_state |= exceptions;
  return GOC_SUCCESS;
}

} // namespace

int goc_v_div_fmas_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *c, uint32_t condition, uint32_t *trapsts) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, true>(flags, exec_mask, instruction_flags, d, a, b, c, condition, trapsts);
}

int goc_v_div_fmas_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *c, uint32_t condition, uint32_t *trapsts) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, true>(flags, exec_mask, instruction_flags, d, a, b, c, condition, trapsts);
}

int goc_v_div_fmas_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c,
                              uint64_t condition, uint32_t *trapsts) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run_wave64<32, true>(flags, exec_mask, instruction_flags, d, a, b, c, condition, trapsts);
}

int goc_v_div_fmas_f64_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c,
                              uint64_t condition, uint32_t *trapsts) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run_wave64<64, true>(flags, exec_mask, instruction_flags, d, a, b, c, condition, trapsts);
}

int goc_v_div_fmas_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c, uint32_t condition,
                             uint32_t *excp_flag_user) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32, false>(flags, exec_mask, instruction_flags, d, a, b, c, condition, excp_flag_user);
}

int goc_v_div_fmas_f64_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c, uint32_t condition,
                             uint32_t *excp_flag_user) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, false>(flags, exec_mask, instruction_flags, d, a, b, c, condition, excp_flag_user);
}

int goc_v_div_fmas_f32_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c,
                                    uint64_t condition, uint32_t *excp_flag_user) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run_wave64<32, false>(flags, exec_mask, instruction_flags, d, a, b, c, condition,
                               excp_flag_user);
}

int goc_v_div_fmas_f64_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c,
                                    uint64_t condition, uint32_t *excp_flag_user) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run_wave64<64, false>(flags, exec_mask, instruction_flags, d, a, b, c, condition,
                               excp_flag_user);
}

// SPDX-License-Identifier: MIT

#include "packed_fma.h"
#include "goc/goc.h"
#include "half_fma_scalar.h"
#include "internal.h"
#include "packed_alu.h"

#include <stdint.h>

namespace {

static const uint64_t fp_flags =
    GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS | GOC_FP_IEEE_MODE;

template <bool Rdna3, bool Accumulate = false>
int run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
        uint32_t *excp_flag_user) {
  bool report = excp_flag_user && (flags & GOC_SEMANTICS_MASK) == GOC_SEMANTICS_EXACT_EMPIRICAL;
  if (report && (Rdna3 || (flags & (GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS))))
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  constexpr uint32_t allowed = Accumulate ? GOC_ALU_ABS_A | GOC_ALU_ABS_B | GOC_ALU_NEG_A |
                                                GOC_ALU_NEG_B | GOC_ALU_CLAMP | GOC_ALU_OMOD_HALF
                                          : 0x1fffU;
  if (int error = goc::validate(flags, mode & ~allowed, true, fp_flags))
    return error;
  if (exec_mask == 0)
    return GOC_SUCCESS;
  uint32_t modes[2];
  if constexpr (Accumulate) {
    // Packed F16 ignores OMOD on both RDNA3 and RDNA4. ABS/NEG apply to
    // both A/B components; the destination accumulator is unmodified.
    modes[0] = mode & ~GOC_ALU_OMOD_HALF;
    modes[1] = modes[0] | GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_C;
  } else {
    modes[0] = goc::packed_half_mode(mode, false);
    modes[1] = goc::packed_half_mode(mode, true);
  }
#if defined(GOC_HAVE_X86_64_V3)
  bool exact = (flags & GOC_SEMANTICS_MASK) == GOC_SEMANTICS_EXACT_EMPIRICAL;
  if (!exact && (flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::packed_fma_x86_64_v3<Accumulate>(flags, exec_mask, mode, d[0], a[0], b[0], c[0]);
    return GOC_SUCCESS;
  }
#endif
  uint32_t exceptions = 0;
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    result[lane] = 0;
    for (int half = 0; half < 2; ++half) {
      uint32_t m = modes[half];
      uint16_t x = uint16_t(a[0][lane] >> (m & GOC_ALU_HIGH_A ? 16 : 0));
      uint16_t y = uint16_t(b[0][lane] >> (m & GOC_ALU_HIGH_B ? 16 : 0));
      uint16_t z = uint16_t(c[0][lane] >> (m & GOC_ALU_HIGH_C ? 16 : 0));
      result[lane] |=
          uint32_t(goc::half_fma_value<Rdna3>(
              x, y, z, m, flags, report && ((exec_mask >> lane) & 1) ? &exceptions : nullptr))
          << (16 * half);
    }
  }
  for (int lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = result[lane];
  if (report)
    *excp_flag_user |= exceptions;
  return GOC_SUCCESS;
}

template <bool Rdna3, bool Accumulate = false>
int run_wave64(uint64_t flags, uint64_t exec_mask, uint32_t mode, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
               uint32_t *excp_flag_user) {
  if (int error = run<Rdna3, Accumulate>(flags, 0, mode, nullptr, nullptr, nullptr, nullptr,
                                         excp_flag_user))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
  bool report = excp_flag_user && (flags & GOC_SEMANTICS_MASK) == GOC_SEMANTICS_EXACT_EMPIRICAL;
  uint32_t exceptions = 0;
  for (unsigned half = 0; half < 2; ++half) {
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *x = a[0] + half * 32, *y = b[0] + half * 32, *z = c[0] + half * 32;
    if (int error = run<Rdna3, Accumulate>(flags, uint32_t(exec_mask >> (half * 32)), mode, &dest,
                                           &x, &y, &z, report ? &exceptions : nullptr))
      return error;
  }
  // Delay the scalar output so it may alias any source, including the upper half.
  if (report)
    *excp_flag_user |= exceptions;
  return GOC_SUCCESS;
}

} // namespace

int goc_v_pk_fma_f16_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<false>(flags, exec_mask, mode, d, a, b, c, excp_flag_user);
}

int goc_v_pk_fmac_f16_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<false, true>(flags, exec_mask, mode, d, a, b, d, excp_flag_user);
}

int goc_v_pk_fma_f16_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c,
                                  uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run_wave64<false>(flags, exec_mask, mode, d, a, b, c, excp_flag_user);
}

int goc_v_pk_fmac_f16_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run_wave64<false, true>(flags, exec_mask, mode, d, a, b, d, excp_flag_user);
}

int goc_v_pk_fma_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
                     uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<true>(flags, exec_mask, mode, d, a, b, c, trapsts);
}

int goc_v_pk_fmac_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                      const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<true, true>(flags, exec_mask, mode, d, a, b, d, trapsts);
}

int goc_v_pk_fma_f16_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run_wave64<true>(flags, exec_mask, mode, d, a, b, c, trapsts);
}

int goc_v_pk_fmac_f16_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run_wave64<true, true>(flags, exec_mask, mode, d, a, b, d, trapsts);
}

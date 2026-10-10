// SPDX-License-Identifier: MIT

#include "trig.h"
#include "dpp.h"
#include "goc/goc.h"
#include "internal.h"
#include "trig_exceptions.h"
#include "trig_model.h"

#include <stdint.h>

namespace {

// Scale a finite trig result by an exact power of two with RNE underflow.
// Input magnitude is at most one; no multiplication can overflow.
template <bool Rdna3> uint32_t output(uint32_t bits, uint32_t mode) {
  uint32_t magnitude = bits & 0x7fffffff;
  if constexpr (Rdna3) {
    // Borrowed apply_omod_f32: flush before scaling, retaining the sign when
    // halving a normal result into the subnormal range.
    unsigned scale = (mode >> 6) & 3;
    if (scale && magnitude < 0x7f800000) {
      if (magnitude < 0x800000)
        bits = 0;
      else if (scale == 3 && magnitude < 0x1000000)
        bits &= 0x80000000;
      else
        bits += scale == 3 ? -0x800000U : scale * 0x800000U;
    }
    magnitude = bits & 0x7fffffff;
  } else if (magnitude < 0x7f800000) {
    unsigned scale = (mode >> 6) & 3;
    if (scale == 3) {
      magnitude =
          magnitude < 0x1000000 ? (magnitude >> 1) + ((magnitude & 3) == 3) : magnitude - 0x800000;
    } else {
      for (unsigned i = 0; i < scale; ++i)
        magnitude = magnitude < 0x800000 ? magnitude * 2 : magnitude + 0x800000;
    }
    bits = (bits & 0x80000000) | magnitude;
  }
  if (mode & GOC_ALU_CLAMP) {
    if ((bits >> 31) || magnitude > 0x7f800000)
      return 0;
    if (magnitude > 0x3f800000)
      return 0x3f800000;
  }
  if constexpr (!Rdna3)
    if ((mode & GOC_ALU_OMOD_HALF) && (bits & 0x7fffffff) < 0x800000)
      return 0;
  return bits;
}

// Compute loose results with guest denormal and NaN policies, consuming all
// inputs before writing any destination.
#if defined(GOC_HAVE_X86_64_V3)
template <bool Rdna3, bool Cosine>
[[gnu::noinline]] void loose_policy(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                                    const uint32_t *a) {
  uint32_t input[32], result[32];
  for (unsigned lane = 0; lane < 32; ++lane) {
    uint32_t bits = a[lane];
    if (mode & GOC_ALU_ABS_A)
      bits &= 0x7fffffff;
    if (mode & GOC_ALU_NEG_A)
      bits ^= 0x80000000;
    if ((flags & GOC_FP_FLUSH_INPUT_DENORMALS) && (bits & 0x7fffffff) < 0x800000)
      bits &= 0x80000000;
    input[lane] = bits;
  }
  goc::trig_x86_64_v3(Cosine, UINT32_MAX, 0, result, input);
  for (unsigned lane = 0; lane < 32; ++lane) {
    uint32_t bits = result[lane];
    if constexpr (Rdna3)
      if ((input[lane] & 0x7fffffff) > 0x7f800000)
        bits = input[lane] | (flags & GOC_FP_IEEE_MODE ? 0x400000 : 0);
    if ((flags & GOC_FP_FLUSH_OUTPUT_DENORMALS) && (bits & 0x7fffffff) < 0x800000)
      bits &= 0x80000000;
    if ((exec_mask >> lane) & 1)
      d[lane] = output<Rdna3>(bits, mode);
  }
}

#endif

template <bool Cosine, bool Rdna3>
int trig(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
         const uint32_t *const *a, uint32_t *excp_flag_user) {
  const bool report = excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE;
  if (report &&
      (Rdna3 || (flags & (GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS)) ||
       (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL))
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return goc::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return trig<Cosine, Rdna3>(flags, exec_mask, uint32_t(mode), d, source, excp_flag_user);
        });
  const uint32_t known = GOC_ALU_NEG_A | GOC_ALU_ABS_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP;
  if (int error = goc::validate(flags, mode & ~known, true,
                                GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS |
                                    GOC_FP_IEEE_MODE))
    return error;
  if constexpr (Rdna3)
    if (flags & GOC_FP_IEEE_MODE)
      mode &= ~uint64_t(GOC_ALU_OMOD_HALF);
  if (!exec_mask)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3 &&
      (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_EXACT_EMPIRICAL) {
    if (Rdna3 || (flags & (GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS)))
      loose_policy<Rdna3, Cosine>(flags, exec_mask, mode, d[0], a[0]);
    else
      goc::trig_x86_64_v3(Cosine, exec_mask, mode, d[0], a[0]);
    return GOC_SUCCESS;
  }
#endif
  uint32_t exceptions = 0;
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    uint32_t bits = a[0][lane];
    if (mode & GOC_ALU_ABS_A)
      bits &= 0x7fffffff;
    if (mode & GOC_ALU_NEG_A)
      bits ^= 0x80000000;
    unsigned denorm = 3 ^ (flags & GOC_FP_FLUSH_INPUT_DENORMALS ? 1 : 0) ^
                      (flags & GOC_FP_FLUSH_OUTPUT_DENORMALS ? 2 : 0);
    uint32_t rounded =
        goc::trig::evaluate(bits, Cosine, denorm, !Rdna3 || (flags & GOC_FP_IEEE_MODE));
    result[lane] = output<Rdna3>(rounded, mode);
    if (report && ((exec_mask >> lane) & 1))
      exceptions |= goc::trig::exceptions<32>(bits, rounded, Cosine, mode);
  }
  for (int lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = result[lane];
  if (report)
    *excp_flag_user |= exceptions;
  return GOC_SUCCESS;
}

template <bool Cosine, bool Rdna3>
int trig_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (int error = trig<Cosine, Rdna3>(flags, 0, mode, nullptr, nullptr, excp_flag_user))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
  bool report = excp_flag_user && (flags & GOC_SEMANTICS_MASK) == GOC_SEMANTICS_EXACT_EMPIRICAL;
  // Delay the global-register write until both halves have consumed their inputs.
  uint32_t exceptions = 0;
  for (unsigned half = 0; half < 2; ++half) {
    uint64_t half_mode = mode;
    if (mode & GOC_DPP16) {
      uint64_t rows = (mode & GOC_DPP_ROW_MASK) >> GOC_DPP_ROW_SHIFT;
      half_mode = (mode & ~GOC_DPP_ROW_MASK) | (((rows >> (half * 2)) & 3) << GOC_DPP_ROW_SHIFT);
    }
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *source = a[0] + half * 32;
    if (int error = trig<Cosine, Rdna3>(flags, uint32_t(exec_mask >> (half * 32)), half_mode, &dest,
                                        &source, report ? &exceptions : nullptr))
      return error;
  }
  if (report)
    *excp_flag_user |= exceptions;
  return GOC_SUCCESS;
}

} // namespace

int goc_v_sin_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, uint32_t *trapsts) {
  return trig<false, true>(flags, exec_mask, mode, d, a, trapsts);
}

int goc_v_sin_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, uint32_t *trapsts) {
  return trig_wave64<false, true>(flags, exec_mask, mode, d, a, trapsts);
}

int goc_v_sin_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                        const uint32_t *const *a, uint32_t *excp_flag_user) {
  return trig<false, false>(flags, exec_mask, mode, d, a, excp_flag_user);
}

int goc_v_sin_f32_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               uint32_t *excp_flag_user) {
  return trig_wave64<false, false>(flags, exec_mask, mode, d, a, excp_flag_user);
}

int goc_v_cos_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, uint32_t *trapsts) {
  return trig<true, true>(flags, exec_mask, mode, d, a, trapsts);
}

int goc_v_cos_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, uint32_t *trapsts) {
  return trig_wave64<true, true>(flags, exec_mask, mode, d, a, trapsts);
}

int goc_v_cos_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                        const uint32_t *const *a, uint32_t *excp_flag_user) {
  return trig<true, false>(flags, exec_mask, mode, d, a, excp_flag_user);
}

int goc_v_cos_f32_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               uint32_t *excp_flag_user) {
  return trig_wave64<true, false>(flags, exec_mask, mode, d, a, excp_flag_user);
}

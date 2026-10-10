// SPDX-License-Identifier: MIT

#include "dpp.h"
#include "float_formats.h"
#include "goc/goc.h"
#include "internal.h"
#include "trig.h"
#include "trig_exceptions.h"
#include "trig_model.h"

#include <stdint.h>

namespace {

// Apply output modifiers to an already rounded FP16 trig result. Active OMOD
// flushes tiny inputs before scaling and tiny outputs after RNE narrowing.
uint16_t output(uint16_t bits, uint32_t mode) {
  uint32_t magnitude = bits & 0x7fff;
  unsigned scale = (mode >> 6) & 3;
  if (scale && magnitude < 0x7c00) {
    if (magnitude < 0x400)
      bits = 0;
    else {
      if (scale == 3)
        magnitude =
            magnitude < 0x800 ? (magnitude >> 1) + ((magnitude & 3) == 3) : magnitude - 0x400;
      else
        magnitude += scale * 0x400;
      bits = magnitude < 0x400 ? 0 : uint16_t((bits & 0x8000) | magnitude);
    }
  }
  if (mode & GOC_ALU_CLAMP) {
    if ((bits & 0x8000) || (bits & 0x7fff) > 0x7c00)
      return 0;
    if (bits > 0x3c00)
      return 0x3c00;
  }
  return bits;
}

#if defined(GOC_HAVE_X86_64_V3)
// Compute loose results with guest half-precision policies. Consume all inputs
// before writing selected destination halves; preserve the other halves.
template <bool Rdna3, bool Cosine>
[[gnu::noinline]] void loose_policy(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                                    const uint32_t *a) {
  uint32_t input[32], result[32]{};
  unsigned a_shift = mode & GOC_ALU_HIGH_A ? 16 : 0;
  unsigned d_shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  for (unsigned lane = 0; lane < 32; ++lane) {
    uint16_t bits = uint16_t(a[lane] >> a_shift);
    if (mode & GOC_ALU_ABS_A)
      bits &= 0x7fff;
    if (mode & GOC_ALU_NEG_A)
      bits ^= 0x8000;
    if ((flags & GOC_FP_FLUSH_INPUT_DENORMALS) && (bits & 0x7fff) < 0x400)
      bits &= 0x8000;
    input[lane] = bits;
  }
  goc::half_trig_x86_64_v3(Cosine, UINT32_MAX, 0, result, input);
  for (unsigned lane = 0; lane < 32; ++lane) {
    uint16_t bits = uint16_t(result[lane]);
    if constexpr (Rdna3)
      if ((input[lane] & 0x7fff) > 0x7c00)
        bits = uint16_t(input[lane] | (flags & GOC_FP_IEEE_MODE ? 0x200 : 0));
    if ((flags & GOC_FP_FLUSH_OUTPUT_DENORMALS) && (bits & 0x7fff) < 0x400)
      bits &= 0x8000;
    if ((exec_mask >> lane) & 1)
      d[lane] = (d[lane] & ~(0xffffU << d_shift)) | (uint32_t(output(bits, mode)) << d_shift);
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
  const uint32_t known = GOC_ALU_NEG_A | GOC_ALU_ABS_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP |
                         GOC_ALU_HIGH_A | GOC_ALU_HIGH_D;
  if (int error = goc::validate(flags, mode & ~known, true,
                                GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS |
                                    GOC_FP_IEEE_MODE))
    return error;
  if constexpr (Rdna3)
    if (!(flags & GOC_FP_FLUSH_OUTPUT_DENORMALS) || (flags & GOC_FP_IEEE_MODE))
      mode &= ~uint64_t(GOC_ALU_OMOD_HALF);
  if (!exec_mask)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3 &&
      (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_EXACT_EMPIRICAL) {
    if (Rdna3 || (flags & (GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS)))
      loose_policy<Rdna3, Cosine>(flags, exec_mask, mode, d[0], a[0]);
    else
      goc::half_trig_x86_64_v3(Cosine, exec_mask, mode, d[0], a[0]);
    return GOC_SUCCESS;
  }
#endif
  const int a_shift = mode & GOC_ALU_HIGH_A ? 16 : 0;
  const int d_shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  uint32_t exceptions = 0;
  uint16_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    uint16_t half = uint16_t(a[0][lane] >> a_shift);
    if (mode & GOC_ALU_ABS_A)
      half &= 0x7fff;
    if (mode & GOC_ALU_NEG_A)
      half ^= 0x8000;
    if ((flags & GOC_FP_FLUSH_INPUT_DENORMALS) && (half & 0x7fff) < 0x400)
      half &= 0x8000;
    uint32_t bits = goc::trig::evaluate(goc::as_bits(goc::f16_to_float(half)), Cosine, 3,
                                        !Rdna3 || (flags & GOC_FP_IEEE_MODE));
    uint16_t rounded = goc::float_to_f16(goc::as_float(bits));
    if ((flags & GOC_FP_FLUSH_OUTPUT_DENORMALS) && (rounded & 0x7fff) < 0x400)
      rounded &= 0x8000;
    result[lane] = output(rounded, mode);
    if (report && ((exec_mask >> lane) & 1))
      exceptions |= goc::trig::exceptions<16>(half, rounded, Cosine, mode);
  }
  for (int lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = (d[0][lane] & ~(0xffffU << d_shift)) | (uint32_t(result[lane]) << d_shift);
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

int goc_v_sin_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, uint32_t *trapsts) {
  return trig<false, true>(flags, exec_mask, mode, d, a, trapsts);
}

int goc_v_sin_f16_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, uint32_t *trapsts) {
  return trig_wave64<false, true>(flags, exec_mask, mode, d, a, trapsts);
}

int goc_v_sin_f16_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                        const uint32_t *const *a, uint32_t *excp_flag_user) {
  return trig<false, false>(flags, exec_mask, mode, d, a, excp_flag_user);
}

int goc_v_sin_f16_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               uint32_t *excp_flag_user) {
  return trig_wave64<false, false>(flags, exec_mask, mode, d, a, excp_flag_user);
}

int goc_v_cos_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, uint32_t *trapsts) {
  return trig<true, true>(flags, exec_mask, mode, d, a, trapsts);
}

int goc_v_cos_f16_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, uint32_t *trapsts) {
  return trig_wave64<true, true>(flags, exec_mask, mode, d, a, trapsts);
}

int goc_v_cos_f16_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                        const uint32_t *const *a, uint32_t *excp_flag_user) {
  return trig<true, false>(flags, exec_mask, mode, d, a, excp_flag_user);
}

int goc_v_cos_f16_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               uint32_t *excp_flag_user) {
  return trig_wave64<true, false>(flags, exec_mask, mode, d, a, excp_flag_user);
}

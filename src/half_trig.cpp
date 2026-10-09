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

template <bool Cosine>
int trig(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
         const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (mode >> 32)
    return goc::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return trig<Cosine>(flags, exec_mask, uint32_t(mode), d, source, excp_flag_user);
        });
  const uint32_t known = GOC_ALU_NEG_A | GOC_ALU_ABS_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP |
                         GOC_ALU_HIGH_A | GOC_ALU_HIGH_D;
  if (int error = goc::validate(flags, mode & ~known, true))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3 &&
      (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_EXACT_EMPIRICAL) {
    goc::half_trig_x86_64_v3(Cosine, exec_mask, mode, d[0], a[0]);
    return GOC_SUCCESS;
  }
#endif
  const int a_shift = mode & GOC_ALU_HIGH_A ? 16 : 0;
  const int d_shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  bool report = excp_flag_user && (flags & GOC_SEMANTICS_MASK) == GOC_SEMANTICS_EXACT_EMPIRICAL;
  uint32_t exceptions = 0;
  uint16_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    uint16_t half = uint16_t(a[0][lane] >> a_shift);
    if (mode & GOC_ALU_ABS_A)
      half &= 0x7fff;
    if (mode & GOC_ALU_NEG_A)
      half ^= 0x8000;
    uint32_t bits = goc::trig::evaluate(goc::as_bits(goc::f16_to_float(half)), Cosine);
    uint16_t rounded = goc::float_to_f16(goc::as_float(bits));
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

} // namespace

int goc_v_sin_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return trig<false>(flags, exec_mask, mode, d, a, excp_flag_user);
}

int goc_v_cos_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return trig<true>(flags, exec_mask, mode, d, a, excp_flag_user);
}

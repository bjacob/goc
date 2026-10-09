// SPDX-License-Identifier: MIT

#include "mixed_fma.h"
#include "dpp.h"
#include "float_formats.h"
#include "goc/goc.h"
#include "half_fma_scalar.h"
#include "internal.h"
#include "mixed_fma_scalar.h"

#include <cmath>
#include <stdint.h>

namespace {

template <goc::MixedFma Dst>
int mixed_fma(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
              const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (mode >> 32)
    return goc::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t filtered, const uint32_t *const *source) {
          return mixed_fma<Dst>(flags, filtered, uint32_t(mode), d, source, b, c);
        });
  const uint32_t known = GOC_ALU_NEG_A | GOC_ALU_NEG_B | GOC_ALU_NEG_C | GOC_ALU_ABS_A |
                         GOC_ALU_ABS_B | GOC_ALU_ABS_C | GOC_ALU_CLAMP | GOC_ALU_HIGH_A |
                         GOC_ALU_HIGH_B | GOC_ALU_HIGH_C | GOC_MIX_F16_A | GOC_MIX_F16_B |
                         GOC_MIX_F16_C;
  if (int error = goc::validate(flags, mode & ~known, Dst != goc::MixedFma::Float))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V3)
  bool exact =
      Dst != goc::MixedFma::Float && (flags & GOC_SEMANTICS_MASK) == GOC_SEMANTICS_EXACT_EMPIRICAL;
  if (!exact && (flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    if constexpr (Dst == goc::MixedFma::Float)
      goc::mixed_fma_float_x86_64_v3(exec_mask, mode, d[0], a[0], b[0], c[0]);
    else
      goc::mixed_fma_half_x86_64_v3(Dst == goc::MixedFma::High, flags & GOC_FP16_OVFL, exec_mask,
                                    mode, d[0], a[0], b[0], c[0]);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    uint32_t x = goc::mixed_fma_input(a[0][lane], mode),
             y = goc::mixed_fma_input(b[0][lane], mode >> 1),
             z = goc::mixed_fma_input(c[0][lane], mode >> 2);
    if constexpr (Dst == goc::MixedFma::Float) {
      float value = std::fma(goc::as_float(x), goc::as_float(y), goc::as_float(z));
      if (mode & GOC_ALU_CLAMP)
        value = !(value > 0.0f) ? 0.0f : value > 1.0f ? 1.0f : value;
      result[lane] = goc::as_bits(value);
    } else {
      constexpr unsigned shift = Dst == goc::MixedFma::High ? 16 : 0;
      uint16_t half =
          goc::mixed_fma_half_value(x, y, z, mode & GOC_ALU_CLAMP, flags & GOC_FP16_OVFL);
      result[lane] = (d[0][lane] & ~(uint32_t(65535) << shift)) | (uint32_t(half) << shift);
    }
  }
  for (int lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_v_fma_mix_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                      const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return mixed_fma<goc::MixedFma::Float>(flags, exec_mask, mode, d, a, b, c);
}

// RX 9070 MIXLO/MIXHI do not generate exception flags, including invalid
// arithmetic and overflow. A requested update therefore leaves the register
// unchanged; the numerical path and its validation are unchanged.
int goc_v_fma_mixlo_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return mixed_fma<goc::MixedFma::Low>(flags, exec_mask, mode, d, a, b, c);
}

int goc_v_fma_mixhi_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return mixed_fma<goc::MixedFma::High>(flags, exec_mask, mode, d, a, b, c);
}

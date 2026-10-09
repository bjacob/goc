// SPDX-License-Identifier: MIT

#include "rdna4_ldexp.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_alu.h"
#include "rdna4_dpp.h"
#include "rdna4_fp64.h"

#include <cmath>
#include <cstring>
#include <stdint.h>

namespace {

template <bool Fp64>
int ldexp(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
          const uint32_t *const *a, const uint32_t *const *b) {
  const uint32_t known = GOC_ALU_NEG_A | GOC_ALU_ABS_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP;
  if (int error = goc::validate(flags, mode & ~known))
    return error;
  if (uint32_t(mask) == 0)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::ldexp_x86_64_v4(Fp64, uint32_t(mask), mode, d, a, b[0]);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::ldexp_x86_64_v3(Fp64, uint32_t(mask), mode, d, a, b[0]);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[Fp64 ? 2 : 1][32];
  for (int lane = 0; lane < 32; ++lane) {
    int32_t exponent;
    std::memcpy(&exponent, b[0] + lane, sizeof(exponent));
    if constexpr (Fp64) {
      double input = goc::fp64_input(a, lane, mode);
      double value = std::ldexp(input, exponent);
      if ((mode & GOC_ALU_OMOD_HALF) && std::abs(value) == 0x1p-1022) {
        // A tiny exact result can round up to minimum normal. Detect this
        // from the source exponent without relying on wider host FP types.
        int source_exponent;
        std::frexp(input, &source_exponent);
        if (int64_t(source_exponent) + exponent <= -1022)
          value = 0;
      }
      uint64_t bits = goc::double_bits(goc::fp64_output(value, mode));
      result[0][lane] = uint32_t(bits);
      result[1][lane] = uint32_t(bits >> 32);
    } else {
      float input = goc::alu_input(a[0][lane], mode);
      float value;
      if (mode & GOC_ALU_OMOD_HALF) {
        // OMOD flushes tininess before FP32 rounding, including values
        // that would round upward to minimum normal.
        double wide = std::ldexp(double(input), exponent);
        value = std::abs(wide) < 0x1p-126 ? 0.0f : float(wide);
      } else {
        value = std::ldexp(input, exponent);
      }
      result[0][lane] = goc::as_bits(goc::alu_output_f32(value, mode));
    }
  }
  for (int reg = 0; reg < (Fp64 ? 2 : 1); ++reg)
    for (int lane = 0; lane < 32; ++lane)
      if ((mask >> lane) & 1)
        d[reg][lane] = result[reg][lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_ldexp_f32(uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b) {
  if (mode >> 32)
    return goc::execute_dpp(flags, mask, mode, a,
                            [&](uint32_t effective, const uint32_t *const *source) {
                              return ldexp<false>(flags, effective, uint32_t(mode), d, source, b);
                            });
  return ldexp<false>(flags, mask, mode, d, a, b);
}

int goc_rdna4_v_ldexp_f64(uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b) {
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return ldexp<true>(flags, mask, mode, d, a, b);
}

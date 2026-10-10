// SPDX-License-Identifier: MIT

#include "fp8_narrow.h"
#include "dpp.h"
#include "goc/goc.h"
#include "internal.h"

#include <stdint.h>

namespace {

template <bool Bf8, bool Stochastic>
int convert(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b) {
  if (mode >> 32)
    return goc::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return convert<Bf8, Stochastic>(flags, exec_mask, uint32_t(mode), d, source, b);
        });
  const uint32_t known =
      GOC_ALU_ABS_A | GOC_ALU_NEG_A |
      (Stochastic ? GOC_CVT_BYTE_3 : GOC_ALU_ABS_B | GOC_ALU_NEG_B | GOC_ALU_HIGH_D);
  if (int error = goc::validate(flags, mode & ~known, true))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
  bool saturate = flags & GOC_FP16_OVFL;
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::fp8_narrow_x86_64_v4<Bf8, Stochastic>(exec_mask, mode, saturate, d[0], a[0], b[0]);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::fp8_narrow_x86_64_v3<Bf8, Stochastic>(exec_mask, mode, saturate, d[0], a[0], b[0]);
    return GOC_SUCCESS;
  }
#endif
  unsigned shift = Stochastic ? ((mode >> 16) & 3) * 8 : mode & GOC_ALU_HIGH_D ? 16 : 0;
  uint32_t selected = (Stochastic ? 255u : 65535u) << shift;
  uint32_t result[32];
  for (unsigned lane = 0; lane < 32; ++lane) {
    uint32_t av = a[0][lane], bv = b[0][lane];
    av &= mode & GOC_ALU_ABS_A ? 0x7fffffff : UINT32_MAX;
    av ^= mode & GOC_ALU_NEG_A ? 0x80000000 : 0;
    uint32_t value = goc::narrow_fp8<Bf8, Stochastic>(av, bv, saturate);
    if constexpr (!Stochastic) {
      bv &= mode & GOC_ALU_ABS_B ? 0x7fffffff : UINT32_MAX;
      bv ^= mode & GOC_ALU_NEG_B ? 0x80000000 : 0;
      value |= goc::narrow_fp8<Bf8, false>(bv, 0, saturate) << 8;
    }
    result[lane] = (d[0][lane] & ~selected) | (value << shift);
  }
  for (unsigned lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

template <bool Bf8, bool Stochastic>
int convert_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a, const uint32_t *const *b) {
  if (int error = convert<Bf8, Stochastic>(flags, 0, mode, nullptr, nullptr, nullptr))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
  for (unsigned half = 0; half < 2; ++half) {
    uint64_t half_mode = mode;
    if (mode & GOC_DPP16) {
      uint64_t rows = (mode & GOC_DPP_ROW_MASK) >> GOC_DPP_ROW_SHIFT;
      half_mode = (mode & ~GOC_DPP_ROW_MASK) | (((rows >> (half * 2)) & 3) << GOC_DPP_ROW_SHIFT);
    }
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *source_a = a[0] + half * 32, *source_b = b[0] + half * 32;
    if (int error = convert<Bf8, Stochastic>(flags, uint32_t(exec_mask >> (half * 32)), half_mode,
                                             &dest, &source_a, &source_b))
      return error;
  }
  return GOC_SUCCESS;
}

} // namespace

int goc_v_cvt_pk_fp8_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b) {
  return convert<false, false>(flags, exec_mask, mode, d, a, b);
}

int goc_v_cvt_pk_bf8_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b) {
  return convert<true, false>(flags, exec_mask, mode, d, a, b);
}

int goc_v_cvt_sr_fp8_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b) {
  return convert<false, true>(flags, exec_mask, mode, d, a, b);
}

int goc_v_cvt_sr_bf8_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b) {
  return convert<true, true>(flags, exec_mask, mode, d, a, b);
}

int goc_v_cvt_pk_fp8_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b) {
  return convert_wave64<false, false>(flags, exec_mask, mode, d, a, b);
}

int goc_v_cvt_pk_bf8_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b) {
  return convert_wave64<true, false>(flags, exec_mask, mode, d, a, b);
}

int goc_v_cvt_sr_fp8_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b) {
  return convert_wave64<false, true>(flags, exec_mask, mode, d, a, b);
}

int goc_v_cvt_sr_bf8_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b) {
  return convert_wave64<true, true>(flags, exec_mask, mode, d, a, b);
}

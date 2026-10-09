// SPDX-License-Identifier: MIT

#include "rdna4_fp8_conversion.h"
#include "float_formats.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_dpp.h"

#include <stdint.h>

namespace {

template <bool Bf8, bool Packed>
int convert(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a) {
  if (mode >> 32) {
    if constexpr (Packed)
      return GOC_ERROR_INVALID_FLAGS;
    else
      return goc::execute_dpp(
          flags, mask, mode, a, [&](uint32_t effective, const uint32_t *const *source) {
            return convert<Bf8, Packed>(flags, effective, uint32_t(mode), d, source);
          });
  }
  const uint32_t known = Packed ? GOC_ALU_HIGH_A : GOC_CVT_BYTE_3;
  if (int error = goc::validate(flags, mode & ~known))
    return error;
  if (!mask)
    return GOC_SUCCESS;
  unsigned shift = Packed ? (mode & GOC_ALU_HIGH_A ? 16 : 0) : 8 * (mode >> 16);
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::fp8_conversion_x86_64_v4<Bf8, Packed>(mask, shift, d, a[0]);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::fp8_conversion_x86_64_v3<Bf8, Packed>(mask, shift, d, a[0]);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[Packed ? 2 : 1][32];
  for (unsigned lane = 0; lane < 32; ++lane) {
    uint32_t raw = a[0][lane] >> shift;
    result[0][lane] = goc::as_bits(goc::fp8_to_float<Bf8>(uint8_t(raw)));
    if constexpr (Packed)
      result[1][lane] = goc::as_bits(goc::fp8_to_float<Bf8>(uint8_t(raw >> 8)));
  }
  for (unsigned reg = 0; reg < (Packed ? 2u : 1u); ++reg)
    for (unsigned lane = 0; lane < 32; ++lane)
      if ((mask >> lane) & 1)
        d[reg][lane] = result[reg][lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_cvt_f32_fp8(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a) {
  return convert<false, false>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_cvt_f32_bf8(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a) {
  return convert<true, false>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_cvt_pk_f32_fp8(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a) {
  return convert<false, true>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_cvt_pk_f32_bf8(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a) {
  return convert<true, true>(flags, exec_mask, instruction_flags, d, a);
}

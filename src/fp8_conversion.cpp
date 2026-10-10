// SPDX-License-Identifier: MIT

#include "fp8_conversion.h"
#include "dpp.h"
#include "float_formats.h"
#include "goc/goc.h"
#include "internal.h"

#include <stdint.h>

namespace {

template <bool Bf8, bool Packed>
int convert(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a) {
  if (mode >> 32) {
    if constexpr (Packed)
      return GOC_ERROR_INVALID_FLAGS;
    else
      return goc::execute_dpp(
          flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
            return convert<Bf8, Packed>(flags, exec_mask, uint32_t(mode), d, source);
          });
  }
  const uint32_t known = Packed ? GOC_ALU_HIGH_A : GOC_CVT_BYTE_3;
  if (int error = goc::validate(flags, mode & ~known, true))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
  unsigned shift = Packed ? (mode & GOC_ALU_HIGH_A ? 16 : 0) : 8 * (mode >> 16);
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::fp8_conversion_x86_64_v4<Bf8, Packed>(exec_mask, shift, d, a[0]);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::fp8_conversion_x86_64_v3<Bf8, Packed>(exec_mask, shift, d, a[0]);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[Packed ? 2 : 1][32];
  for (unsigned lane = 0; lane < 32; ++lane) {
    uint32_t raw = a[0][lane] >> shift;
    // RX 9070 widening canonicalizes either NaN sign to the negative quiet NaN.
    auto widen = [](uint8_t code) {
      unsigned magnitude = code & 127;
      if (Bf8 ? magnitude > 124 : magnitude == 127)
        return 0xffc00000U;
      return goc::as_bits(goc::fp8_to_float<Bf8>(code));
    };
    result[0][lane] = widen(uint8_t(raw));
    if constexpr (Packed)
      result[1][lane] = widen(uint8_t(raw >> 8));
  }
  for (unsigned reg = 0; reg < (Packed ? 2u : 1u); ++reg)
    for (unsigned lane = 0; lane < 32; ++lane)
      if ((exec_mask >> lane) & 1)
        d[reg][lane] = result[reg][lane];
  return GOC_SUCCESS;
}

template <bool Bf8, bool Packed>
int convert_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a) {
  if (int error = convert<Bf8, Packed>(flags, 0, mode, nullptr, nullptr))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
  for (unsigned half = 0; half < 2; ++half) {
    uint64_t half_mode = mode;
    if (mode & GOC_DPP16) {
      uint64_t rows = (mode & GOC_DPP_ROW_MASK) >> GOC_DPP_ROW_SHIFT;
      half_mode = (mode & ~GOC_DPP_ROW_MASK) | (((rows >> (half * 2)) & 3) << GOC_DPP_ROW_SHIFT);
    }
    uint32_t *dest[Packed ? 2 : 1];
    for (unsigned reg = 0; reg < (Packed ? 2U : 1U); ++reg)
      dest[reg] = d[reg] + half * 32;
    const uint32_t *source = a[0] + half * 32;
    if (int error = convert<Bf8, Packed>(flags, uint32_t(exec_mask >> (half * 32)), half_mode, dest,
                                         &source))
      return error;
  }
  return GOC_SUCCESS;
}

} // namespace

int goc_v_cvt_f32_fp8(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a) {
  return convert<false, false>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_cvt_f32_bf8(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                      uint32_t *const *d, const uint32_t *const *a) {
  return convert<true, false>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_cvt_pk_f32_fp8(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a) {
  return convert<false, true>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_cvt_pk_f32_bf8(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a) {
  return convert<true, true>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_cvt_f32_fp8_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a) {
  return convert_wave64<false, false>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_cvt_f32_bf8_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a) {
  return convert_wave64<true, false>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_cvt_pk_f32_fp8_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a) {
  return convert_wave64<false, true>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_cvt_pk_f32_bf8_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a) {
  return convert_wave64<true, true>(flags, exec_mask, instruction_flags, d, a);
}

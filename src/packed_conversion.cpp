// SPDX-License-Identifier: MIT

#include "packed_conversion.h"
#include "dpp.h"
#include "goc/goc.h"
#include "half_conversion.h"
#include "internal.h"

#include <cmath>
#include <stdint.h>

namespace {

template <goc::PackedConversion Op> uint16_t narrow(uint32_t raw, uint32_t mode) {
  if (mode & GOC_ALU_ABS_A)
    raw &= 0x7fffffff;
  if (mode & GOC_ALU_NEG_A)
    raw ^= 0x80000000;
  if constexpr (Op == goc::PackedConversion::HalfRtz)
    return goc::half_rtz(raw);
  else {
    float value = goc::as_float(raw);
    if constexpr (Op == goc::PackedConversion::Unsigned)
      return !(value > 0) ? 0 : value >= 65535 ? 65535 : uint16_t(value);
    else
      return std::isnan(value) ? 0
             : value >= 32767  ? 32767
             : value <= -32768 ? 32768
                               : uint16_t(int16_t(value));
  }
}

template <goc::PackedConversion Op>
int convert(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b) {
  if (mode >> 32)
    return goc::execute_dpp(flags, exec_mask, mode, a,
                            [&](uint32_t exec_mask, const uint32_t *const *source) {
                              return convert<Op>(flags, exec_mask, uint32_t(mode), d, source, b);
                            });
  const uint32_t known = GOC_ALU_ABS_A | GOC_ALU_ABS_B | GOC_ALU_NEG_A | GOC_ALU_NEG_B |
                         GOC_ALU_CLAMP |
                         (Op == goc::PackedConversion::HalfRtz ? GOC_ALU_OMOD_HALF : 0);
  if (int error = goc::validate(flags, mode & ~known, false,
                                GOC_FP_IEEE_MODE | GOC_FP_FLUSH_INPUT_DENORMALS |
                                    GOC_FP_FLUSH_OUTPUT_DENORMALS))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::packed_conversion_x86_64_v4<Op>(exec_mask, mode, d[0], a[0], b[0]);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::packed_conversion_x86_64_v3<Op>(exec_mask, mode, d[0], a[0], b[0]);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (unsigned lane = 0; lane < 32; ++lane)
    result[lane] =
        narrow<Op>(a[0][lane], mode) | (uint32_t(narrow<Op>(b[0][lane], mode >> 1)) << 16);
  for (unsigned lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

template <goc::PackedConversion Op>
int convert_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a, const uint32_t *const *b) {
  if (int error = convert<Op>(flags, 0, mode, nullptr, nullptr, nullptr))
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
    if (int error = convert<Op>(flags, uint32_t(exec_mask >> (half * 32)), half_mode, &dest,
                                &source_a, &source_b))
      return error;
  }
  return GOC_SUCCESS;
}

} // namespace

int goc_v_cvt_pk_rtz_f16_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return convert<goc::PackedConversion::HalfRtz>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_v_cvt_pk_i16_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return convert<goc::PackedConversion::Signed>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_v_cvt_pk_u16_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return convert<goc::PackedConversion::Unsigned>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_v_cvt_pk_rtz_f16_f32_rdna4_wave64(uint64_t flags, uint64_t exec_mask,
                                          uint64_t instruction_flags, uint32_t *const *d,
                                          const uint32_t *const *a, const uint32_t *const *b,
                                          uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return convert_wave64<goc::PackedConversion::HalfRtz>(flags, exec_mask, instruction_flags, d, a,
                                                        b);
}

int goc_v_cvt_pk_i16_f32_rdna4_wave64(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return convert_wave64<goc::PackedConversion::Signed>(flags, exec_mask, instruction_flags, d, a,
                                                       b);
}

int goc_v_cvt_pk_u16_f32_rdna4_wave64(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return convert_wave64<goc::PackedConversion::Unsigned>(flags, exec_mask, instruction_flags, d, a,
                                                         b);
}

int goc_v_cvt_pk_rtz_f16_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return goc_v_cvt_pk_rtz_f16_f32_rdna4(flags, exec_mask, mode, d, a, b, nullptr);
}

int goc_v_cvt_pk_rtz_f16_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return goc_v_cvt_pk_rtz_f16_f32_rdna4_wave64(flags, exec_mask, mode, d, a, b, nullptr);
}

int goc_v_cvt_pk_i16_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return goc_v_cvt_pk_i16_f32_rdna4(flags, exec_mask, mode, d, a, b, nullptr);
}

int goc_v_cvt_pk_i16_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return goc_v_cvt_pk_i16_f32_rdna4_wave64(flags, exec_mask, mode, d, a, b, nullptr);
}

int goc_v_cvt_pk_u16_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return goc_v_cvt_pk_u16_f32_rdna4(flags, exec_mask, mode, d, a, b, nullptr);
}

int goc_v_cvt_pk_u16_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return goc_v_cvt_pk_u16_f32_rdna4_wave64(flags, exec_mask, mode, d, a, b, nullptr);
}

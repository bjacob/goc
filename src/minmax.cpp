// SPDX-License-Identifier: MIT

#include "minmax.h"
#include "alu.h"
#include "dpp.h"
#include "goc/goc.h"
#include "internal.h"

#include <stdint.h>

namespace {

template <goc::Minmax3 Op, bool FirstMaximum, bool SecondMaximum, bool Propagate,
          bool Rdna3 = false>
int minmax3(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (mode >> 32) {
    return goc::execute_dpp(flags, exec_mask, mode, a,
                            [&](uint32_t exec_mask, const uint32_t *const *source) {
                              return minmax3<Op, FirstMaximum, SecondMaximum, Propagate, Rdna3>(
                                  flags, exec_mask, uint32_t(mode), d, source, b, c);
                            });
  }

  constexpr uint64_t fp_flags =
      GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS | GOC_FP_IEEE_MODE;
  if (int error = goc::validate(flags, mode & ~0x1ffU, false, fp_flags))
    return error;
  if constexpr (Rdna3)
    if (!(flags & GOC_FP_FLUSH_OUTPUT_DENORMALS) || (flags & GOC_FP_IEEE_MODE))
      mode &= ~uint64_t(GOC_ALU_OMOD_HALF);
  if (exec_mask == 0)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::minmax3_x86_64_v3(Op, flags, exec_mask, mode, d[0], a[0], b[0], c[0]);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    float x = goc::alu_input(a[0][lane], mode);
    float y = goc::alu_input(b[0][lane], mode >> 1);
    float z = goc::alu_input(c[0][lane], mode >> 2);
    if (flags & GOC_FP_FLUSH_INPUT_DENORMALS) {
      x = goc::flush_denorm_f32(x);
      y = goc::flush_denorm_f32(y);
      z = goc::flush_denorm_f32(z);
    }
    float value;
    if constexpr (Rdna3)
      value = goc::minmax3_rdna3<FirstMaximum, SecondMaximum, Op == goc::Minmax3::MedianLegacy>(
          x, y, z, flags & GOC_FP_IEEE_MODE);
    else
      value =
          goc::minmax3_value<FirstMaximum, SecondMaximum, Propagate, Op == goc::Minmax3::MedianNum>(
              x, y, z);
    result[lane] = goc::as_bits(goc::alu_output_f32(value, mode));
  }
  for (int lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

template <goc::Minmax3 Op, bool FirstMaximum, bool SecondMaximum, bool Propagate,
          bool Rdna3 = false>
int minmax3_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (int error = minmax3<Op, FirstMaximum, SecondMaximum, Propagate, Rdna3>(
          flags, 0, mode, nullptr, nullptr, nullptr, nullptr))
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
    const uint32_t *x = a[0] + half * 32, *y = b[0] + half * 32, *z = c[0] + half * 32;
    if (int error = minmax3<Op, FirstMaximum, SecondMaximum, Propagate, Rdna3>(
            flags, uint32_t(exec_mask >> (half * 32)), half_mode, &dest, &x, &y, &z))
      return error;
  }
  return GOC_SUCCESS;
}

} // namespace

int goc_v_min3_num_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                       const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
                       uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return minmax3<goc::Minmax3::Min3Num, false, false, false>(flags, exec_mask, mode, d, a, b, c);
}

int goc_v_max3_num_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                       const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
                       uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return minmax3<goc::Minmax3::Max3Num, true, true, false>(flags, exec_mask, mode, d, a, b, c);
}

int goc_v_minmax_num_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return minmax3<goc::Minmax3::MinmaxNum, false, true, false>(flags, exec_mask, mode, d, a, b, c);
}

int goc_v_maxmin_num_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return minmax3<goc::Minmax3::MaxminNum, true, false, false>(flags, exec_mask, mode, d, a, b, c);
}

int goc_v_minimum3_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                       const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
                       uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return minmax3<goc::Minmax3::Minimum3, false, false, true>(flags, exec_mask, mode, d, a, b, c);
}

int goc_v_maximum3_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                       const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
                       uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return minmax3<goc::Minmax3::Maximum3, true, true, true>(flags, exec_mask, mode, d, a, b, c);
}

int goc_v_minimummaximum_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return minmax3<goc::Minmax3::MinimumMaximum, false, true, true>(flags, exec_mask, mode, d, a, b,
                                                                  c);
}

int goc_v_maximumminimum_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return minmax3<goc::Minmax3::MaximumMinimum, true, false, true>(flags, exec_mask, mode, d, a, b,
                                                                  c);
}

int goc_v_med3_num_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                       const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
                       uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return minmax3<goc::Minmax3::MedianNum, false, false, false>(flags, exec_mask, mode, d, a, b, c);
}

int goc_v_min3_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
                   uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return minmax3<goc::Minmax3::Min3Legacy, false, false, false, true>(flags, exec_mask, mode, d, a,
                                                                      b, c);
}

int goc_v_max3_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
                   uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return minmax3<goc::Minmax3::Max3Legacy, true, true, false, true>(flags, exec_mask, mode, d, a, b,
                                                                    c);
}

int goc_v_minmax_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
                     uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return minmax3<goc::Minmax3::MinmaxLegacy, false, true, false, true>(flags, exec_mask, mode, d, a,
                                                                       b, c);
}

int goc_v_maxmin_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
                     uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return minmax3<goc::Minmax3::MaxminLegacy, true, false, false, true>(flags, exec_mask, mode, d, a,
                                                                       b, c);
}

int goc_v_med3_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
                   uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return minmax3<goc::Minmax3::MedianLegacy, false, false, false, true>(flags, exec_mask, mode, d,
                                                                        a, b, c);
}

int goc_v_min3_num_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return minmax3_wave64<goc::Minmax3::Min3Num, false, false, false>(flags, exec_mask, mode, d, a, b,
                                                                    c);
}

int goc_v_max3_num_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return minmax3_wave64<goc::Minmax3::Max3Num, true, true, false>(flags, exec_mask, mode, d, a, b,
                                                                  c);
}

int goc_v_minmax_num_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return minmax3_wave64<goc::Minmax3::MinmaxNum, false, true, false>(flags, exec_mask, mode, d, a,
                                                                     b, c);
}

int goc_v_maxmin_num_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return minmax3_wave64<goc::Minmax3::MaxminNum, true, false, false>(flags, exec_mask, mode, d, a,
                                                                     b, c);
}

int goc_v_minimum3_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return minmax3_wave64<goc::Minmax3::Minimum3, false, false, true>(flags, exec_mask, mode, d, a, b,
                                                                    c);
}

int goc_v_maximum3_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return minmax3_wave64<goc::Minmax3::Maximum3, true, true, true>(flags, exec_mask, mode, d, a, b,
                                                                  c);
}

int goc_v_minimummaximum_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c,
                                    uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return minmax3_wave64<goc::Minmax3::MinimumMaximum, false, true, true>(flags, exec_mask, mode, d,
                                                                         a, b, c);
}

int goc_v_maximumminimum_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c,
                                    uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return minmax3_wave64<goc::Minmax3::MaximumMinimum, true, false, true>(flags, exec_mask, mode, d,
                                                                         a, b, c);
}

int goc_v_med3_num_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return minmax3_wave64<goc::Minmax3::MedianNum, false, false, false>(flags, exec_mask, mode, d, a,
                                                                      b, c);
}

int goc_v_min3_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *c, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return minmax3_wave64<goc::Minmax3::Min3Legacy, false, false, false, true>(flags, exec_mask, mode,
                                                                             d, a, b, c);
}

int goc_v_max3_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *c, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return minmax3_wave64<goc::Minmax3::Max3Legacy, true, true, false, true>(flags, exec_mask, mode,
                                                                           d, a, b, c);
}

int goc_v_minmax_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return minmax3_wave64<goc::Minmax3::MinmaxLegacy, false, true, false, true>(flags, exec_mask,
                                                                              mode, d, a, b, c);
}

int goc_v_maxmin_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return minmax3_wave64<goc::Minmax3::MaxminLegacy, true, false, false, true>(flags, exec_mask,
                                                                              mode, d, a, b, c);
}

int goc_v_med3_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *c, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return minmax3_wave64<goc::Minmax3::MedianLegacy, false, false, false, true>(flags, exec_mask,
                                                                               mode, d, a, b, c);
}

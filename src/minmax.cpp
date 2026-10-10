// SPDX-License-Identifier: MIT

#include "minmax.h"
#include "alu.h"
#include "dpp.h"
#include "goc/goc.h"
#include "internal.h"

#include <stdint.h>

namespace {

template <goc::Minmax3 Op, bool FirstMaximum, bool SecondMaximum, bool Propagate>
int minmax3(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (mode >> 32) {
    return goc::execute_dpp(flags, exec_mask, mode, a,
                            [&](uint32_t exec_mask, const uint32_t *const *source) {
                              return minmax3<Op, FirstMaximum, SecondMaximum, Propagate>(
                                  flags, exec_mask, uint32_t(mode), d, source, b, c);
                            });
  }

  if (int error = goc::validate(flags, mode & ~0x1ffU))
    return error;
  if (exec_mask == 0)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::minmax3_x86_64_v3(Op, exec_mask, mode, d[0], a[0], b[0], c[0]);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    float x = goc::alu_input(a[0][lane], mode);
    float y = goc::alu_input(b[0][lane], mode >> 1);
    float z = goc::alu_input(c[0][lane], mode >> 2);
    float value =
        goc::minmax3_value<FirstMaximum, SecondMaximum, Propagate, Op == goc::Minmax3::MedianNum>(
            x, y, z);
    result[lane] = goc::as_bits(goc::alu_output_f32(value, mode));
  }
  for (int lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = result[lane];
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
                   uint32_t *excp_flag_user) {
  return goc_v_min3_num_f32(flags, exec_mask, mode, d, a, b, c, excp_flag_user);
}

int goc_v_max3_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
                   uint32_t *excp_flag_user) {
  return goc_v_max3_num_f32(flags, exec_mask, mode, d, a, b, c, excp_flag_user);
}

int goc_v_minmax_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
                     uint32_t *excp_flag_user) {
  return goc_v_minmax_num_f32(flags, exec_mask, mode, d, a, b, c, excp_flag_user);
}

int goc_v_maxmin_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
                     uint32_t *excp_flag_user) {
  return goc_v_maxmin_num_f32(flags, exec_mask, mode, d, a, b, c, excp_flag_user);
}

int goc_v_med3_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
                   uint32_t *excp_flag_user) {
  return goc_v_med3_num_f32(flags, exec_mask, mode, d, a, b, c, excp_flag_user);
}

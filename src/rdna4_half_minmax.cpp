// SPDX-License-Identifier: MIT

#include "rdna4_half_minmax.h"
#include "float_formats.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_alu.h"
#include "rdna4_dpp.h"
#include "rdna4_minmax.h"

#include <stdint.h>

namespace {

template <goc::Minmax3 Op, bool FirstMaximum, bool SecondMaximum, bool Propagate>
int minmax3(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (mode >> 32)
    return goc::execute_dpp(flags, exec_mask, mode, a,
                            [&](uint32_t exec_mask, const uint32_t *const *source) {
                              return minmax3<Op, FirstMaximum, SecondMaximum, Propagate>(
                                  flags, exec_mask, uint32_t(mode), d, source, b, c);
                            });
  if (int error = goc::validate(flags, mode & ~0x1fffU))
    return error;
  if (exec_mask == 0)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::half_minmax3_x86_64_v3<FirstMaximum, SecondMaximum, Propagate,
                                Op == goc::Minmax3::MedianNum>(
        bool(flags & GOC_FP16_OVFL), exec_mask, mode, d[0], a[0], b[0], c[0]);
    return GOC_SUCCESS;
  }
#endif
  int a_shift = mode & GOC_ALU_HIGH_A ? 16 : 0;
  int b_shift = mode & GOC_ALU_HIGH_B ? 16 : 0;
  int c_shift = mode & GOC_ALU_HIGH_C ? 16 : 0;
  int d_shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  uint16_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    float x =
        goc::alu_input(goc::as_bits(goc::f16_to_float(uint16_t(a[0][lane] >> a_shift))), mode);
    float y =
        goc::alu_input(goc::as_bits(goc::f16_to_float(uint16_t(b[0][lane] >> b_shift))), mode >> 1);
    float z =
        goc::alu_input(goc::as_bits(goc::f16_to_float(uint16_t(c[0][lane] >> c_shift))), mode >> 2);
    float value = goc::minmax3_value<FirstMaximum, SecondMaximum, Propagate,
                                     Op == goc::Minmax3::MedianNum, true>(x, y, z);
    result[lane] = goc::float_to_f16(goc::alu_output_f16(value, mode), flags & GOC_FP16_OVFL);
  }
  for (int lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = (d[0][lane] & ~(0xffffU << d_shift)) | (uint32_t(result[lane]) << d_shift);
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_min3_num_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return minmax3<goc::Minmax3::Min3Num, false, false, false>(flags, exec_mask, mode, d, a, b, c);
}

int goc_rdna4_v_max3_num_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return minmax3<goc::Minmax3::Max3Num, true, true, false>(flags, exec_mask, mode, d, a, b, c);
}

int goc_rdna4_v_minmax_num_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c,
                               uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return minmax3<goc::Minmax3::MinmaxNum, false, true, false>(flags, exec_mask, mode, d, a, b, c);
}

int goc_rdna4_v_maxmin_num_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c,
                               uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return minmax3<goc::Minmax3::MaxminNum, true, false, false>(flags, exec_mask, mode, d, a, b, c);
}

int goc_rdna4_v_minimum3_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return minmax3<goc::Minmax3::Minimum3, false, false, true>(flags, exec_mask, mode, d, a, b, c);
}

int goc_rdna4_v_maximum3_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return minmax3<goc::Minmax3::Maximum3, true, true, true>(flags, exec_mask, mode, d, a, b, c);
}

int goc_rdna4_v_minimummaximum_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, const uint32_t *const *c,
                                   uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return minmax3<goc::Minmax3::MinimumMaximum, false, true, true>(flags, exec_mask, mode, d, a, b,
                                                                  c);
}

int goc_rdna4_v_maximumminimum_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, const uint32_t *const *c,
                                   uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return minmax3<goc::Minmax3::MaximumMinimum, true, false, true>(flags, exec_mask, mode, d, a, b,
                                                                  c);
}

int goc_rdna4_v_med3_num_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return minmax3<goc::Minmax3::MedianNum, false, false, false>(flags, exec_mask, mode, d, a, b, c);
}

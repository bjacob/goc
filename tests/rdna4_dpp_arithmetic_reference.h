// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"
#include "internal.h"
#include "rdna4_binary_reference.h"
#include "rdna4_dpp16_reference.h"
#include "rdna4_minmax_reference.h"

#include <stdint.h>

namespace goc_test {

inline const char *const dpp_arithmetic_names[] = {"v_add_f32",
                                                   "v_sub_f32",
                                                   "v_subrev_f32",
                                                   "v_mul_f32",
                                                   "v_min_num_f32",
                                                   "v_max_num_f32",
                                                   "v_minimum_f32",
                                                   "v_maximum_f32",
                                                   "v_mul_dx9_zero_f32",
                                                   "v_min3_num_f32",
                                                   "v_max3_num_f32",
                                                   "v_minmax_num_f32",
                                                   "v_maxmin_num_f32",
                                                   "v_minimum3_f32",
                                                   "v_maximum3_f32",
                                                   "v_minimummaximum_f32",
                                                   "v_maximumminimum_f32",
                                                   "v_med3_num_f32"};

inline int dpp_arithmetic_call(unsigned op, uint64_t flags, uint64_t mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c) {
  using Binary = decltype(&goc_rdna4_v_add_f32);
  using Ternary = decltype(&goc_rdna4_v_min3_num_f32);
  static const Binary binary[] = {
      goc_rdna4_v_add_f32,     goc_rdna4_v_sub_f32,     goc_rdna4_v_subrev_f32,
      goc_rdna4_v_mul_f32,     goc_rdna4_v_min_num_f32, goc_rdna4_v_max_num_f32,
      goc_rdna4_v_minimum_f32, goc_rdna4_v_maximum_f32, goc_rdna4_v_mul_dx9_zero_f32};
  static const Ternary ternary[] = {
      goc_rdna4_v_min3_num_f32,       goc_rdna4_v_max3_num_f32,       goc_rdna4_v_minmax_num_f32,
      goc_rdna4_v_maxmin_num_f32,     goc_rdna4_v_minimum3_f32,       goc_rdna4_v_maximum3_f32,
      goc_rdna4_v_minimummaximum_f32, goc_rdna4_v_maximumminimum_f32, goc_rdna4_v_med3_num_f32};
  return op < 9 ? binary[op](flags, mask, mode, d, a, b)
                : ternary[op - 9](flags, mask, mode, d, a, b, c);
}

inline const uint64_t dpp_arithmetic_modes[] = {
    UINT64_C(0x539770100000000), UINT64_C(0x539770300000000), UINT64_C(0x1ff010400000000),
    UINT64_C(0x1ff010c00000000), UINT64_C(0x1ff010600000000), UINT64_C(0x1ff010e00000000),
    UINT64_C(0xa3670c00000000)};

inline uint32_t dpp_arithmetic_reference(unsigned op, uint32_t a, uint32_t b, uint32_t c,
                                         uint32_t low) {
  return op < 9 ? goc::as_bits(binary_reference(op, goc::as_float(a), goc::as_float(b), low))
                : minmax_reference::reference(op - 9, a, b, c, low);
}

inline bool dpp_arithmetic_source(uint64_t mode, uint32_t mask, unsigned lane, int &source) {
  if (mode & GOC_DPP8) {
    unsigned selectors = unsigned(mode >> GOC_DPP8_SELECT_SHIFT);
    source = int(lane / 8) * 8 + int((selectors >> (3 * (lane % 8))) & 7);
    if (!(mode & GOC_DPP_FI) && !(mask & (1u << source)))
      source = -1;
    return (mask & (1u << lane)) != 0;
  }
  return dpp16_reference(
      unsigned((mode & GOC_DPP_CTRL_MASK) >> GOC_DPP_CTRL_SHIFT), mode & GOC_DPP_FI,
      mode & GOC_DPP_BOUND_CTRL, unsigned((mode & GOC_DPP_ROW_MASK) >> GOC_DPP_ROW_SHIFT),
      unsigned((mode & GOC_DPP_BANK_MASK) >> GOC_DPP_BANK_SHIFT), mask, lane, source);
}

} // namespace goc_test

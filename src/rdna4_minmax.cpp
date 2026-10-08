// SPDX-License-Identifier: MIT

#include "rdna4_minmax.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_alu.h"

#include <stdint.h>

namespace {

template <goc::Minmax3 Op, bool FirstMaximum, bool SecondMaximum, bool Propagate>
int minmax3(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (int error = goc::validate(flags, mode & ~UINT32_C(0x1ff)))
    return error;
  if (uint32_t(mask) == 0)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::minmax3_x86_64_v3(Op, uint32_t(mask), mode, d[0], a[0], b[0], c[0]);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    float x = goc::alu_input(a[0][lane], mode);
    float y = goc::alu_input(b[0][lane], mode >> 1);
    float z = goc::alu_input(c[0][lane], mode >> 2);
    float ab = goc::minmax<FirstMaximum, Propagate>(x, y);
    float value = goc::minmax<SecondMaximum, Propagate>(ab, z);
    result[lane] = goc::as_bits(goc::alu_output(value, mode));
  }
  for (int lane = 0; lane < 32; ++lane)
    if ((mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_min3_num_f32(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c) {
  return minmax3<goc::Minmax3::Min3Num, false, false, false>(flags, mask, mode, d, a, b, c);
}

int goc_rdna4_v_max3_num_f32(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c) {
  return minmax3<goc::Minmax3::Max3Num, true, true, false>(flags, mask, mode, d, a, b, c);
}

int goc_rdna4_v_minmax_num_f32(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               const uint32_t *const *c) {
  return minmax3<goc::Minmax3::MinmaxNum, false, true, false>(flags, mask, mode, d, a, b, c);
}

int goc_rdna4_v_maxmin_num_f32(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               const uint32_t *const *c) {
  return minmax3<goc::Minmax3::MaxminNum, true, false, false>(flags, mask, mode, d, a, b, c);
}

int goc_rdna4_v_minimum3_f32(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c) {
  return minmax3<goc::Minmax3::Minimum3, false, false, true>(flags, mask, mode, d, a, b, c);
}

int goc_rdna4_v_maximum3_f32(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c) {
  return minmax3<goc::Minmax3::Maximum3, true, true, true>(flags, mask, mode, d, a, b, c);
}

int goc_rdna4_v_minimummaximum_f32(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                                   const uint32_t *const *a, const uint32_t *const *b,
                                   const uint32_t *const *c) {
  return minmax3<goc::Minmax3::MinimumMaximum, false, true, true>(flags, mask, mode, d, a, b, c);
}

int goc_rdna4_v_maximumminimum_f32(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                                   const uint32_t *const *a, const uint32_t *const *b,
                                   const uint32_t *const *c) {
  return minmax3<goc::Minmax3::MaximumMinimum, true, false, true>(flags, mask, mode, d, a, b, c);
}

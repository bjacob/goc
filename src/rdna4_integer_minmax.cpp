// SPDX-License-Identifier: MIT

#include "rdna4_integer_minmax.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_dpp.h"

#include <algorithm>
#include <stdint.h>

namespace {

template <goc::IntegerMinmax Op, bool Signed>
int minmax(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
           const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (mode >> 32)
    return goc::execute_dpp(
        flags, mask, mode, a, [&](uint32_t effective, const uint32_t *const *source) {
          return minmax<Op, Signed>(flags, effective, uint32_t(mode), d, source, b, c);
        });

  if (int error = goc::validate(flags, mode))
    return error;
  if (mask == 0)
    return GOC_SUCCESS;
  constexpr bool binary = Op == goc::IntegerMinmax::Min || Op == goc::IntegerMinmax::Max;
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::integer_minmax_x86_64_v4<Op, Signed>(mask, d[0], a[0], b[0], binary ? nullptr : c[0]);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::integer_minmax_x86_64_v3<Op, Signed>(mask, d[0], a[0], b[0], binary ? nullptr : c[0]);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[32];
  // Bias signed values into unsigned ordering, avoiding implementation-defined
  // uint32_t-to-int32_t conversions while preserving the selected input bits.
  const uint32_t bias = Signed ? UINT32_C(0x80000000) : 0;
  for (int lane = 0; lane < 32; ++lane) {
    uint32_t x = a[0][lane] ^ bias, y = b[0][lane] ^ bias, z = 0, value;
    if constexpr (!binary)
      z = c[0][lane] ^ bias;
    if constexpr (Op == goc::IntegerMinmax::Min)
      value = std::min(x, y);
    if constexpr (Op == goc::IntegerMinmax::Max)
      value = std::max(x, y);
    if constexpr (Op == goc::IntegerMinmax::Min3)
      value = std::min(std::min(x, y), z);
    if constexpr (Op == goc::IntegerMinmax::Max3)
      value = std::max(std::max(x, y), z);
    // Mixed operations combine A/B first, as specified by the RDNA4 ISA.
    if constexpr (Op == goc::IntegerMinmax::Minmax)
      value = std::max(std::min(x, y), z);
    if constexpr (Op == goc::IntegerMinmax::Maxmin)
      value = std::min(std::max(x, y), z);
    // The same median selection network is used by rocjitsu vector_alu.py.
    if constexpr (Op == goc::IntegerMinmax::Median)
      value = std::max(std::min(std::max(x, y), z), std::min(x, y));
    result[lane] = value ^ bias;
  }
  for (int lane = 0; lane < 32; ++lane)
    if (mask >> lane & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_min_i32(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b) {
  return minmax<goc::IntegerMinmax::Min, true>(flags, mask, mode, d, a, b, nullptr);
}

int goc_rdna4_v_max_i32(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b) {
  return minmax<goc::IntegerMinmax::Max, true>(flags, mask, mode, d, a, b, nullptr);
}

int goc_rdna4_v_min3_i32(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c) {
  return minmax<goc::IntegerMinmax::Min3, true>(flags, mask, mode, d, a, b, c);
}

int goc_rdna4_v_max3_i32(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c) {
  return minmax<goc::IntegerMinmax::Max3, true>(flags, mask, mode, d, a, b, c);
}

int goc_rdna4_v_minmax_i32(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  return minmax<goc::IntegerMinmax::Minmax, true>(flags, mask, mode, d, a, b, c);
}

int goc_rdna4_v_maxmin_i32(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  return minmax<goc::IntegerMinmax::Maxmin, true>(flags, mask, mode, d, a, b, c);
}

int goc_rdna4_v_med3_i32(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c) {
  return minmax<goc::IntegerMinmax::Median, true>(flags, mask, mode, d, a, b, c);
}

int goc_rdna4_v_min_u32(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b) {
  return minmax<goc::IntegerMinmax::Min, false>(flags, mask, mode, d, a, b, nullptr);
}

int goc_rdna4_v_max_u32(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b) {
  return minmax<goc::IntegerMinmax::Max, false>(flags, mask, mode, d, a, b, nullptr);
}

int goc_rdna4_v_min3_u32(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c) {
  return minmax<goc::IntegerMinmax::Min3, false>(flags, mask, mode, d, a, b, c);
}

int goc_rdna4_v_max3_u32(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c) {
  return minmax<goc::IntegerMinmax::Max3, false>(flags, mask, mode, d, a, b, c);
}

int goc_rdna4_v_minmax_u32(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  return minmax<goc::IntegerMinmax::Minmax, false>(flags, mask, mode, d, a, b, c);
}

int goc_rdna4_v_maxmin_u32(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  return minmax<goc::IntegerMinmax::Maxmin, false>(flags, mask, mode, d, a, b, c);
}

int goc_rdna4_v_med3_u32(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c) {
  return minmax<goc::IntegerMinmax::Median, false>(flags, mask, mode, d, a, b, c);
}

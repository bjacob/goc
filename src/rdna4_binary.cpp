// SPDX-License-Identifier: MIT

#include "rdna4_binary.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_alu.h"

#include <stdint.h>

namespace {

// IEEE selection orders -0 below +0. Number variants ignore even signaling NaNs
// when the other operand is numeric; propagating variants prefer signaling NaNs.
template <bool Maximum, bool Propagate> float minmax(float x, float y) {
  uint32_t a = goc::as_bits(x), b = goc::as_bits(y);
  bool an = (a & 0x7fffffff) > 0x7f800000, bn = (b & 0x7fffffff) > 0x7f800000;
  if constexpr (Propagate) {
    if (an && !(a & 0x00400000))
      return goc::as_float(a | 0x00400000);
    if (bn && !(b & 0x00400000))
      return goc::as_float(b | 0x00400000);
    if (an || bn)
      return goc::as_float((an ? a : b) | 0x00400000);
  } else {
    if (an)
      return bn ? goc::as_float(a | 0x00400000) : y;
    if (bn)
      return x;
  }
  if (x == y)
    return goc::as_float(Maximum ? (a & b) : (a | b));
  return (Maximum ? x > y : x < y) ? x : y;
}

template <goc::Binary Op>
int binary(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
           const uint32_t *const *a, const uint32_t *const *b) {
  const uint32_t known = GOC_ALU_NEG_A | GOC_ALU_NEG_B | GOC_ALU_ABS_A | GOC_ALU_ABS_B |
                         GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP;
  if (int error = goc::validate(flags, mode & ~known))
    return error;
  if (uint32_t(mask) == 0)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::binary_x86_64_v3(Op, uint32_t(mask), mode, d[0], a[0], b[0]);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    float x = goc::alu_input(a[0][lane], mode), y = goc::alu_input(b[0][lane], mode >> 1);
    float value;
    if constexpr (Op == goc::Binary::Add)
      value = x + y;
    if constexpr (Op == goc::Binary::Sub)
      value = x - y;
    if constexpr (Op == goc::Binary::Subrev)
      value = y - x;
    if constexpr (Op == goc::Binary::Mul)
      value = x * y;
    if constexpr (Op == goc::Binary::MinNum)
      value = minmax<false, false>(x, y);
    if constexpr (Op == goc::Binary::MaxNum)
      value = minmax<true, false>(x, y);
    if constexpr (Op == goc::Binary::Minimum)
      value = minmax<false, true>(x, y);
    if constexpr (Op == goc::Binary::Maximum)
      value = minmax<true, true>(x, y);
    result[lane] = goc::as_bits(goc::alu_output(value, mode));
  }
  for (int lane = 0; lane < 32; ++lane)
    if ((mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_add_f32(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b) {
  return binary<goc::Binary::Add>(flags, mask, mode, d, a, b);
}

int goc_rdna4_v_sub_f32(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b) {
  return binary<goc::Binary::Sub>(flags, mask, mode, d, a, b);
}

int goc_rdna4_v_subrev_f32(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b) {
  return binary<goc::Binary::Subrev>(flags, mask, mode, d, a, b);
}

int goc_rdna4_v_mul_f32(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b) {
  return binary<goc::Binary::Mul>(flags, mask, mode, d, a, b);
}

int goc_rdna4_v_min_num_f32(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b) {
  return binary<goc::Binary::MinNum>(flags, mask, mode, d, a, b);
}

int goc_rdna4_v_max_num_f32(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b) {
  return binary<goc::Binary::MaxNum>(flags, mask, mode, d, a, b);
}

int goc_rdna4_v_minimum_f32(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b) {
  return binary<goc::Binary::Minimum>(flags, mask, mode, d, a, b);
}

int goc_rdna4_v_maximum_f32(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b) {
  return binary<goc::Binary::Maximum>(flags, mask, mode, d, a, b);
}

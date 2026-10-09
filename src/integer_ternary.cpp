// SPDX-License-Identifier: MIT

#include "integer_ternary.h"
#include "dpp.h"
#include "goc/goc.h"
#include "internal.h"

#include <stdint.h>

namespace {

template <goc::IntegerTernary Op> uint32_t evaluate(uint32_t a, uint32_t b, uint32_t c) {
  if constexpr (Op == goc::IntegerTernary::ShiftAdd)
    return (a << (b & 31)) + c;
  if constexpr (Op == goc::IntegerTernary::AddShift)
    return (a + b) << (c & 31);
  if constexpr (Op == goc::IntegerTernary::ShiftOr)
    return (a << (b & 31)) | c;
  if constexpr (Op == goc::IntegerTernary::AndOr)
    return (a & b) | c;
  if constexpr (Op == goc::IntegerTernary::Or3)
    return a | b | c;
  if constexpr (Op == goc::IntegerTernary::Xor3)
    return a ^ b ^ c;
  if constexpr (Op == goc::IntegerTernary::XorAdd)
    return (a ^ b) + c;
  if constexpr (Op == goc::IntegerTernary::Lerp) {
    // Per-byte rounding follows rocjitsu's vector_alu.py LERP implementation.
    uint32_t result = 0;
    for (int byte = 0; byte < 4; ++byte) {
      unsigned shift = unsigned(8 * byte);
      unsigned sum = ((a >> shift) & 255) + ((b >> shift) & 255) + ((c >> shift) & 1);
      result |= (sum >> 1) << shift;
    }
    return result;
  }
}

template <goc::IntegerTernary Op>
int ternary(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (mode >> 32)
    return goc::execute_dpp(flags, exec_mask, mode, a,
                            [&](uint32_t exec_mask, const uint32_t *const *source) {
                              return ternary<Op>(flags, exec_mask, uint32_t(mode), d, source, b, c);
                            });

  if (int error = goc::validate(flags, mode))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::integer_ternary_x86_64_v4<Op>(exec_mask, d[0], a[0], b[0], c[0]);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  // The simple Boolean/XOR-add variants show no AVX2 gain over baseline.
  if constexpr (goc::integer_ternary_v3_supported(Op)) {
    if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
      goc::integer_ternary_x86_64_v3<Op>(exec_mask, d[0], a[0], b[0], c[0]);
      return GOC_SUCCESS;
    }
  }
#endif
  uint32_t result[32];
  if constexpr (Op == goc::IntegerTernary::ShiftAdd || Op == goc::IntegerTernary::AddShift ||
                Op == goc::IntegerTernary::ShiftOr) {
    // Clang's baseline vector shifts use FP conversions for powers of two;
    // converting 2^31 raises FE_INVALID. Explicit SIMD paths use integer shifts.
#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
#endif
    for (int lane = 0; lane < 32; ++lane)
      result[lane] = evaluate<Op>(a[0][lane], b[0][lane], c[0][lane]);
  } else {
    for (int lane = 0; lane < 32; ++lane)
      result[lane] = evaluate<Op>(a[0][lane], b[0][lane], c[0][lane]);
  }
  for (int lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_v_lshl_add_u32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *c) {
  return ternary<goc::IntegerTernary::ShiftAdd>(flags, exec_mask, mode, d, a, b, c);
}

int goc_v_add_lshl_u32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *c) {
  return ternary<goc::IntegerTernary::AddShift>(flags, exec_mask, mode, d, a, b, c);
}

int goc_v_lshl_or_b32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      const uint32_t *const *c) {
  return ternary<goc::IntegerTernary::ShiftOr>(flags, exec_mask, mode, d, a, b, c);
}

int goc_v_and_or_b32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  return ternary<goc::IntegerTernary::AndOr>(flags, exec_mask, mode, d, a, b, c);
}

int goc_v_or3_b32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  return ternary<goc::IntegerTernary::Or3>(flags, exec_mask, mode, d, a, b, c);
}

int goc_v_xor3_b32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  return ternary<goc::IntegerTernary::Xor3>(flags, exec_mask, mode, d, a, b, c);
}

int goc_v_xad_u32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  return ternary<goc::IntegerTernary::XorAdd>(flags, exec_mask, mode, d, a, b, c);
}

int goc_v_lerp_u8(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  return ternary<goc::IntegerTernary::Lerp>(flags, exec_mask, mode, d, a, b, c);
}

// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "float_formats.h"
#include "goc.h"
#include "internal.h"

#include <algorithm>
#include <cmath>
#include <stdint.h>

namespace {

// Adapted from rocjitsu shared/mma_exec.h. RDNA4 wave32 uses eight K
// elements per lane group. K=32 INT4 interleaves groups in eight-element blocks.
template <int Bits, int K = 16> uint32_t element(const uint32_t *const *v, int index, int k) {
  static_assert(K == 16 || (K == 32 && Bits == 4));
  int local = k % 8 + 8 * (k / 16);
  int lane = index + 16 * ((k / 8) % 2);
  return (v[local * Bits / 32][lane] >> ((local * Bits) % 32)) & ((1u << Bits) - 1);
}

void store(uint64_t mask, uint32_t *const *d, const uint32_t (&result)[8][32]) {
  for (int reg = 0; reg < 8; ++reg)
    for (int lane = 0; lane < 32; ++lane)
      if ((mask >> lane) & 1)
        d[reg][lane] = result[reg][lane];
}

template <bool Bf8A, bool Bf8B>
int floating(uint64_t flags, uint64_t mask, uint32_t modifiers, uint32_t *const *d,
             const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  // FP8/BF8 WMMA supports C negation/absolute value, not A/B halfword negation.
  if (int error = goc::validate(flags, modifiers & ~(GOC_WMMA_NEG_C | GOC_WMMA_ABS_C)))
    return error;
  uint32_t result[8][32];
  for (int row = 0; row < 16; ++row)
    for (int col = 0; col < 16; ++col) {
      int lane = col + 16 * (row / 8), reg = row % 8;
      if (!((mask >> lane) & 1))
        continue;
      uint32_t bits = c[reg][lane];
      if (modifiers & GOC_WMMA_ABS_C)
        bits &= 0x7fffffff;
      if (modifiers & GOC_WMMA_NEG_C)
        bits ^= 0x80000000;
      float acc = goc::as_float(bits);
      for (int k = 0; k < 16; ++k)
        acc = std::fma(goc::fp8_to_float<Bf8A>(uint8_t(element<8>(a, row, k))),
                       goc::fp8_to_float<Bf8B>(uint8_t(element<8>(b, col, k))), acc);
      result[reg][lane] = goc::as_bits(acc);
    }
  store(mask, d, result);
  return GOC_SUCCESS;
}

template <int Bits, int K>
int integer(uint64_t flags, uint64_t mask, uint32_t modifiers, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (int error = goc::validate(
          flags, modifiers & ~(GOC_WMMA_SIGNED_A | GOC_WMMA_SIGNED_B | GOC_WMMA_CLAMP), true))
    return error;
  const auto extend = [](uint32_t bits, bool is_signed) -> int64_t {
    return int64_t(bits) - (is_signed && (bits & (1u << (Bits - 1))) ? (1u << Bits) : 0);
  };
  uint32_t result[8][32];
  for (int row = 0; row < 16; ++row)
    for (int col = 0; col < 16; ++col) {
      int lane = col + 16 * (row / 8), reg = row % 8;
      if (!((mask >> lane) & 1))
        continue;
      uint32_t bits = c[reg][lane];
      int64_t acc = int64_t(bits) - ((bits >> 31) ? (INT64_C(1) << 32) : 0);
      for (int k = 0; k < K; ++k)
        acc += extend(element<Bits, K>(a, row, k), modifiers & GOC_WMMA_SIGNED_A) *
               extend(element<Bits, K>(b, col, k), modifiers & GOC_WMMA_SIGNED_B);
      if (modifiers & GOC_WMMA_CLAMP)
        acc = std::clamp(acc, -INT64_C(2147483648), INT64_C(2147483647));
      // Unsigned conversion implements modulo 2^32 without signed overflow.
      result[reg][lane] = uint32_t(acc);
    }
  store(mask, d, result);
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_wmma_f32_16x16x16_fp8_fp8(uint64_t flags, uint64_t mask, uint32_t instruction_flags,
                                          uint32_t *const *d, const uint32_t *const *a,
                                          const uint32_t *const *b, const uint32_t *const *c) {
  return floating<false, false>(flags, mask, instruction_flags, d, a, b, c);
}

int goc_rdna4_v_wmma_f32_16x16x16_fp8_bf8(uint64_t flags, uint64_t mask, uint32_t instruction_flags,
                                          uint32_t *const *d, const uint32_t *const *a,
                                          const uint32_t *const *b, const uint32_t *const *c) {
  return floating<false, true>(flags, mask, instruction_flags, d, a, b, c);
}

int goc_rdna4_v_wmma_f32_16x16x16_bf8_fp8(uint64_t flags, uint64_t mask, uint32_t instruction_flags,
                                          uint32_t *const *d, const uint32_t *const *a,
                                          const uint32_t *const *b, const uint32_t *const *c) {
  return floating<true, false>(flags, mask, instruction_flags, d, a, b, c);
}

int goc_rdna4_v_wmma_f32_16x16x16_bf8_bf8(uint64_t flags, uint64_t mask, uint32_t instruction_flags,
                                          uint32_t *const *d, const uint32_t *const *a,
                                          const uint32_t *const *b, const uint32_t *const *c) {
  return floating<true, true>(flags, mask, instruction_flags, d, a, b, c);
}

int goc_rdna4_v_wmma_i32_16x16x16_iu8(uint64_t flags, uint64_t mask, uint32_t instruction_flags,
                                      uint32_t *const *d, const uint32_t *const *a,
                                      const uint32_t *const *b, const uint32_t *const *c) {
  return integer<8, 16>(flags, mask, instruction_flags, d, a, b, c);
}

int goc_rdna4_v_wmma_i32_16x16x16_iu4(uint64_t flags, uint64_t mask, uint32_t instruction_flags,
                                      uint32_t *const *d, const uint32_t *const *a,
                                      const uint32_t *const *b, const uint32_t *const *c) {
  return integer<4, 16>(flags, mask, instruction_flags, d, a, b, c);
}

int goc_rdna4_v_wmma_i32_16x16x32_iu4(uint64_t flags, uint64_t mask, uint32_t instruction_flags,
                                      uint32_t *const *d, const uint32_t *const *a,
                                      const uint32_t *const *b, const uint32_t *const *c) {
  return integer<4, 32>(flags, mask, instruction_flags, d, a, b, c);
}

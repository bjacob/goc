// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <immintrin.h>
#include <stdint.h>

namespace goc {

template <bool Wide> struct DivisionOpsV4 {
  using V = __m512i;
  using Mask = __mmask16;

  static V set(uint64_t x) {
    if constexpr (Wide)
      return _mm512_set1_epi64(int64_t(x));
    else
      return _mm512_set1_epi32(int32_t(x));
  }

  static V add(V a, V b) {
    if constexpr (Wide)
      return _mm512_add_epi64(a, b);
    else
      return _mm512_add_epi32(a, b);
  }

  static V sub(V a, V b) {
    if constexpr (Wide)
      return _mm512_sub_epi64(a, b);
    else
      return _mm512_sub_epi32(a, b);
  }

  static V shr(V a, int count) {
    if constexpr (Wide)
      return _mm512_srli_epi64(a, count);
    else
      return _mm512_srli_epi32(a, count);
  }

  static V shl(V a, int count) {
    if constexpr (Wide)
      return _mm512_slli_epi64(a, count);
    else
      return _mm512_slli_epi32(a, count);
  }

  static Mask gt(V a, V b) {
    if constexpr (Wide)
      return _mm512_cmp_epi64_mask(a, b, _MM_CMPINT_GT);
    else
      return _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_GT);
  }

  static Mask eq(V a, V b) {
    if constexpr (Wide)
      return _mm512_cmpeq_epi64_mask(a, b);
    else
      return _mm512_cmpeq_epi32_mask(a, b);
  }

  static Mask both(Mask a, Mask b) { return a & b; }

  static Mask either(Mask a, Mask b) { return a | b; }

  static Mask inverse(Mask a) { return Mask(~a); }

  static V select(Mask m, V yes, V no) {
    if constexpr (Wide)
      return _mm512_mask_blend_epi64(__mmask8(m), no, yes);
    else
      return _mm512_mask_blend_epi32(m, no, yes);
  }
};

} // namespace goc

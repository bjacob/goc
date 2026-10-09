// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <immintrin.h>
#include <stdint.h>

namespace goc {

template <bool Wide> struct DivisionOpsV3 {
  using V = __m256i;
  using Mask = __m256i;

  static V set(uint64_t x) {
    if constexpr (Wide)
      return _mm256_set1_epi64x(int64_t(x));
    else
      return _mm256_set1_epi32(int32_t(x));
  }

  static V add(V a, V b) {
    if constexpr (Wide)
      return _mm256_add_epi64(a, b);
    else
      return _mm256_add_epi32(a, b);
  }

  static V sub(V a, V b) {
    if constexpr (Wide)
      return _mm256_sub_epi64(a, b);
    else
      return _mm256_sub_epi32(a, b);
  }

  static V shr(V a, int count) {
    if constexpr (Wide)
      return _mm256_srli_epi64(a, count);
    else
      return _mm256_srli_epi32(a, count);
  }

  static V shl(V a, int count) {
    if constexpr (Wide)
      return _mm256_slli_epi64(a, count);
    else
      return _mm256_slli_epi32(a, count);
  }

  static Mask gt(V a, V b) {
    if constexpr (Wide)
      return _mm256_cmpgt_epi64(a, b);
    else
      return _mm256_cmpgt_epi32(a, b);
  }

  static Mask eq(V a, V b) {
    if constexpr (Wide)
      return _mm256_cmpeq_epi64(a, b);
    else
      return _mm256_cmpeq_epi32(a, b);
  }

  static Mask both(Mask a, Mask b) { return _mm256_and_si256(a, b); }

  static Mask either(Mask a, Mask b) { return _mm256_or_si256(a, b); }

  static Mask inverse(Mask a) { return _mm256_xor_si256(a, _mm256_set1_epi32(-1)); }

  static V select(Mask m, V yes, V no) { return _mm256_blendv_epi8(no, yes, m); }
};

} // namespace goc

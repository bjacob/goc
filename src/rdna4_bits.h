// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc {

// Sign-extend a low-bit field to a 32-bit raw register word.
template <unsigned Bits> uint32_t sign_extend_word(uint32_t value) {
  static_assert(Bits > 0 && Bits <= 32);
  constexpr uint32_t sign = UINT32_C(1) << (Bits - 1);
  return ((value & (UINT32_MAX >> (32 - Bits))) ^ sign) - sign;
}

// Decode the low Bits bits as a signed or unsigned integer, discarding higher bits.
template <unsigned Bits, bool Signed> int64_t extend_integer(uint32_t value) {
  static_assert(Bits > 0 && Bits <= 32);
  value &= UINT32_MAX >> (32 - Bits);
  if constexpr (Signed) {
    constexpr uint32_t sign = UINT32_C(1) << (Bits - 1);
    if constexpr (Bits <= 16)
      return int64_t(value ^ sign) - sign;
    else
      // Keep the wider-field form that avoids slower GCC high-product lowering.
      return int64_t(value) - ((value & sign) ? INT64_C(1) << Bits : 0);
  } else {
    return value;
  }
}

// Number of set bits in an unsigned 32- or 64-bit word.
template <typename T> uint32_t bit_population(T x) {
  x -= (x >> 1) & T(UINT64_C(0x5555555555555555));
  x = (x & T(UINT64_C(0x3333333333333333))) + ((x >> 2) & T(UINT64_C(0x3333333333333333)));
  x = (x + (x >> 4)) & T(UINT64_C(0x0f0f0f0f0f0f0f0f));
  x += x >> 8;
  x += x >> 16;
  if constexpr (sizeof(T) == 8)
    x += x >> 32;
  return uint32_t(x) & (sizeof(T) == 8 ? 127 : 63);
}

// Leading or trailing zero count; a zero word returns UINT32_MAX.
template <bool Trailing, typename T> uint32_t bit_count_zero(T a) {
  if (!a)
    return UINT32_MAX;
#if defined(__GNUC__) || defined(__clang__)
  if constexpr (Trailing) {
    if constexpr (sizeof(T) == 8)
      return __builtin_ctzll(a);
    else
      return __builtin_ctz(a);
  } else {
    if constexpr (sizeof(T) == 8)
      return __builtin_clzll(a);
    else
      return __builtin_clz(a);
  }
#else
  uint32_t n = 0;
  if constexpr (Trailing) {
    while (!(a & 1)) {
      ++n;
      a >>= 1;
    }
  } else {
    const T sign = T(1) << (sizeof(T) * 8 - 1);
    while (!(a & sign)) {
      ++n;
      a <<= 1;
    }
  }
  return n;
#endif
}

// Reverse all bits of an unsigned 32- or 64-bit word.
template <typename T> T bit_reverse(T x) {
  const T m1 = T(UINT64_C(0x5555555555555555));
  const T m2 = T(UINT64_C(0x3333333333333333));
  const T m4 = T(UINT64_C(0x0f0f0f0f0f0f0f0f));
  const T m8 = T(UINT64_C(0x00ff00ff00ff00ff));
  x = ((x >> 1) & m1) | ((x & m1) << 1);
  x = ((x >> 2) & m2) | ((x & m2) << 2);
  x = ((x >> 4) & m4) | ((x & m4) << 4);
  x = ((x >> 8) & m8) | ((x & m8) << 8);
  if constexpr (sizeof(T) == 8) {
    const T m16 = UINT64_C(0x0000ffff0000ffff);
    x = ((x >> 16) & m16) | ((x & m16) << 16);
    return (x >> 32) | (x << 32);
  } else {
    return (x >> 16) | (x << 16);
  }
}

} // namespace goc

// SPDX-License-Identifier: MIT

#pragma once

#include <cstring>
#include <stdint.h>
#include <type_traits>

namespace goc {

// Raw IEEE binary32/binary64 storage and classification masks.
template <typename Float> struct FloatBits {
  static_assert(std::is_same_v<Float, float> || std::is_same_v<Float, double>);
  using UInt = std::conditional_t<std::is_same_v<Float, float>, uint32_t, uint64_t>;
  static constexpr unsigned fraction_bits = std::is_same_v<Float, float> ? 23 : 52;
  static constexpr UInt sign = UInt(1) << (sizeof(UInt) * 8 - 1);
  static constexpr UInt magnitude = sign - 1;
  static constexpr UInt infinity = magnitude & ~((UInt(1) << fraction_bits) - 1);
  static constexpr UInt quiet = UInt(1) << (fraction_bits - 1);

  static UInt bits(Float value) {
    UInt result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
  }

  static Float value(UInt bits) {
    Float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
  }
};

} // namespace goc

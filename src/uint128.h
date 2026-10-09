// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef GOC_UINT128_H_
#define GOC_UINT128_H_

#include <stdint.h>

namespace goc {

// Unsigned modulo-2^128 arithmetic adapted from rocjitsu util/big_int.h.
// Shift counts must be in [0, 127]. Explicit conversions retain the low bits.
class Uint128 {
public:
  constexpr Uint128(uint64_t low = 0, uint64_t high = 0) : low_(low), high_(high) {}

  explicit constexpr operator uint32_t() const { return static_cast<uint32_t>(low_); }

  explicit constexpr operator uint64_t() const { return low_; }

  explicit constexpr operator bool() const { return low_ != 0 || high_ != 0; }

  friend constexpr bool operator==(Uint128 lhs, Uint128 rhs) {
    return lhs.low_ == rhs.low_ && lhs.high_ == rhs.high_;
  }

  friend constexpr bool operator!=(Uint128 lhs, Uint128 rhs) { return !(lhs == rhs); }

  friend constexpr bool operator>(Uint128 lhs, Uint128 rhs) {
    return lhs.high_ == rhs.high_ ? lhs.low_ > rhs.low_ : lhs.high_ > rhs.high_;
  }

  friend constexpr Uint128 operator+(Uint128 lhs, Uint128 rhs) {
    const uint64_t low = lhs.low_ + rhs.low_;
    return {low, lhs.high_ + rhs.high_ + (low < lhs.low_)};
  }

  friend constexpr Uint128 operator-(Uint128 lhs, Uint128 rhs) {
    return {lhs.low_ - rhs.low_, lhs.high_ - rhs.high_ - (lhs.low_ < rhs.low_)};
  }

  friend constexpr Uint128 operator*(Uint128 lhs, Uint128 rhs) {
    // Split the low words into 32-bit limbs so every partial product and carry
    // fits in uint64_t. Terms above bit 127 are discarded, as for unsigned ints.
    const uint64_t a0 = static_cast<uint32_t>(lhs.low_), a1 = lhs.low_ >> 32;
    const uint64_t b0 = static_cast<uint32_t>(rhs.low_), b1 = rhs.low_ >> 32;
    const uint64_t low = a0 * b0;
    const uint64_t cross = a1 * b0 + (low >> 32);
    const uint64_t middle = a0 * b1 + static_cast<uint32_t>(cross);
    const uint64_t high =
        a1 * b1 + (cross >> 32) + (middle >> 32) + lhs.high_ * rhs.low_ + lhs.low_ * rhs.high_;
    return {(middle << 32) | static_cast<uint32_t>(low), high};
  }

  friend constexpr Uint128 operator&(Uint128 lhs, Uint128 rhs) {
    return {lhs.low_ & rhs.low_, lhs.high_ & rhs.high_};
  }

  friend constexpr Uint128 operator|(Uint128 lhs, Uint128 rhs) {
    return {lhs.low_ | rhs.low_, lhs.high_ | rhs.high_};
  }

  friend constexpr Uint128 operator<<(Uint128 value, int distance) {
    if (distance == 0)
      return value;
    if (distance >= 64)
      return {0, value.low_ << (distance - 64)};
    return {value.low_ << distance, (value.high_ << distance) | (value.low_ >> (64 - distance))};
  }

  friend constexpr Uint128 operator>>(Uint128 value, int distance) {
    if (distance == 0)
      return value;
    if (distance >= 64)
      return {value.high_ >> (distance - 64), 0};
    return {(value.low_ >> distance) | (value.high_ << (64 - distance)), value.high_ >> distance};
  }

  constexpr Uint128 &operator++() { return *this = *this + 1; }

  constexpr Uint128 &operator>>=(int distance) { return *this = *this >> distance; }

private:
  uint64_t low_;
  uint64_t high_;
};

} // namespace goc

#endif

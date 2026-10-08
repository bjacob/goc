// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <cstring>
#include <stdint.h>

namespace goc {

enum class Fp64 { Add, Mul, Fma, Trunc, Ceil, Rndne, Floor, Fract, Sqrt, Rcp, Rsq };

constexpr int fp64_sources(Fp64 op) {
  switch (op) {
  case Fp64::Add:
  case Fp64::Mul:
    return 2;
  case Fp64::Fma:
    return 3;
  default:
    return 1;
  }
}

inline double as_double(uint64_t bits) {
  double value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

inline uint64_t double_bits(double value) {
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

// Round to an integral FP64 value, ties to even, preserving signed zero and
// quieting NaNs. Adapted from the rocjitsu-derived FP32 rndne helper.
inline double fp64_rndne(double value) {
  uint64_t raw = double_bits(value), magnitude = raw & UINT64_C(0x7fffffffffffffff);
  uint64_t sign = raw & UINT64_C(0x8000000000000000);
  if (magnitude >= UINT64_C(0x7ff0000000000000))
    return as_double(raw |
                     (magnitude > UINT64_C(0x7ff0000000000000) ? UINT64_C(0x0008000000000000) : 0));
  unsigned exponent = unsigned(magnitude >> 52);
  if (exponent >= 1075)
    return value;
  if (exponent < 1023)
    return as_double(sign |
                     (magnitude > UINT64_C(0x3fe0000000000000) ? UINT64_C(0x3ff0000000000000) : 0));
  uint64_t unit = UINT64_C(1) << (1075 - exponent);
  uint64_t fraction = magnitude & (unit - 1), rounded = magnitude & ~(unit - 1);
  if (fraction > unit / 2 || (fraction == unit / 2 && (rounded & unit)))
    rounded += unit;
  return as_double(sign | rounded);
}

// Each FP64 lane is split into low/high words in two independent VGPR buffers.
inline double fp64_input(const uint32_t *const *v, int lane, uint32_t mode) {
  uint64_t bits = v[0][lane] | (uint64_t(v[1][lane]) << 32);
  if (mode & GOC_ALU_ABS_A)
    bits &= UINT64_C(0x7fffffffffffffff);
  if (mode & GOC_ALU_NEG_A)
    bits ^= UINT64_C(0x8000000000000000);
  return as_double(bits);
}

inline double fp64_output(double value, uint32_t mode) {
  switch ((mode >> 6) & 3) {
  case 1:
    value *= 2;
    break;
  case 2:
    value *= 4;
    break;
  case 3:
    value *= 0.5;
    break;
  }
  if (mode & GOC_ALU_CLAMP)
    value = !(value > 0) ? 0 : value > 1 ? 1 : value;
  return value;
}

#if defined(GOC_HAVE_X86_64_V3)
void fp64_x86_64_v3(Fp64 op, uint32_t mask, uint32_t mode, uint32_t *const *d,
                    const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c);
#endif

} // namespace goc

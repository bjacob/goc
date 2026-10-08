// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <cstring>
#include <stdint.h>

namespace goc {

enum class Fp64 { Add, Mul, Fma };

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

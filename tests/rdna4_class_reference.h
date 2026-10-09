// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <stdint.h>

namespace goc_test {

static const uint16_t class_edges16[] = {0,      0x8000, 1,      0x8001, 0x03ff, 0x83ff,
                                         0x0400, 0x8400, 0x3c00, 0xbc00, 0x7bff, 0xfbff,
                                         0x7c00, 0xfc00, 0x7c01, 0x7e01};
static const uint32_t class_edges32[] = {
    0,          0x80000000, 1,          0x80000001, 0x007fffff, 0x807fffff, 0x00800000, 0x80800000,
    0x3f800000, 0xbf800000, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7f800001, 0x7fc00001};
static const uint64_t class_edges64[] = {0,
                                         0x8000000000000000ULL,
                                         1,
                                         0x8000000000000001ULL,
                                         0xfffffffffffffULL,
                                         0x800fffffffffffffULL,
                                         0x10000000000000ULL,
                                         0x8010000000000000ULL,
                                         0x3ff0000000000000ULL,
                                         0xbff0000000000000ULL,
                                         0x7fefffffffffffffULL,
                                         0xffefffffffffffffULL,
                                         0x7ff0000000000000ULL,
                                         0xfff0000000000000ULL,
                                         0x7ff0000000000001ULL,
                                         0x7ff8000000000001ULL};

inline uint32_t class_mode(unsigned m) {
  return (m & 1 ? GOC_ALU_ABS_A : 0) | (m & 2 ? GOC_ALU_NEG_A : 0) | (m & 4 ? GOC_ALU_HIGH_A : 0) |
         (m & 8 ? GOC_ALU_HIGH_B : 0);
}

inline bool class_reference(unsigned fmt, uint32_t lo, uint32_t hi, uint32_t mask, unsigned m) {
  uint64_t raw = fmt == 2 ? (uint64_t(hi) << 32) | lo : lo;
  unsigned fraction_bits = fmt == 0   ? 10
                           : fmt == 1 ? 23
                                      : 52,
           exponent_bits = fmt == 0   ? 5
                           : fmt == 1 ? 8
                                      : 11;
  if (fmt == 0) {
    raw = (raw >> (m & 4 ? 16 : 0)) & 65535;
    mask >>= m & 8 ? 16 : 0;
  }
  uint64_t sign = 1ULL << (fraction_bits + exponent_bits);
  if (m & 1)
    raw &= ~sign;
  if (m & 2)
    raw ^= sign;
  uint64_t fraction = raw & ((1ULL << fraction_bits) - 1);
  unsigned exponent = unsigned((raw >> fraction_bits) & ((1u << exponent_bits) - 1));
  bool negative = raw & sign, maximum = exponent == ((1u << exponent_bits) - 1);
  bool nan = maximum && fraction, quiet = (raw >> (fraction_bits - 1)) & 1;
  return ((mask & 1) && nan && !quiet) || ((mask & 2) && nan && quiet) ||
         ((mask & (negative ? 4 : 512)) && maximum && !fraction) ||
         ((mask & (negative ? 8 : 256)) && exponent && !maximum) ||
         ((mask & (negative ? 16 : 128)) && !exponent && fraction) ||
         ((mask & (negative ? 32 : 64)) && !exponent && !fraction);
}

inline void class_capture_inputs(unsigned fmt, unsigned start, uint32_t words[3][32]) {
  for (unsigned lane = 0; lane < 32; ++lane) {
    unsigned i = start + lane;
    uint32_t random = (i * 0x7395a831u) ^ 0xa7925163u;
    uint64_t value = fmt == 0    ? uint64_t(i | ((65535 - i) << 16))
                     : fmt == 1  ? uint64_t(i < 32768 ? class_edges32[i % 16] : random)
                     : i < 32768 ? class_edges64[i % 16]
                                 : (uint64_t(random) << 32) | ((i * 0x83a1459du) ^ 0x5389d241u);
    words[0][lane] = uint32_t(value);
    words[1][lane] = uint32_t(value >> 32);
    words[2][lane] = (i * 0x9e3779b9u) ^ 0xa5a59669u;
  }
}

} // namespace goc_test

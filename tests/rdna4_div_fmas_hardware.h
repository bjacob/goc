// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <stdint.h>

namespace goc_test {

// RX 9070 gfx1201, MODE=0xf0, captured 2026-10-09. Cartesian triples, A outermost.
// Columns group by condition bit (clear/set), then the modifier list below.
// FNV-1a consumes little-endian result bytes, four for FP32 and eight for FP64.
static const uint32_t fmas_capture_modes[] = {0,
                                              GOC_ALU_OMOD_2,
                                              GOC_ALU_OMOD_4,
                                              GOC_ALU_OMOD_HALF,
                                              GOC_ALU_CLAMP,
                                              GOC_ALU_NEG_A | GOC_ALU_ABS_A | GOC_ALU_ABS_B |
                                                  GOC_ALU_NEG_C,
                                              GOC_ALU_CLAMP | GOC_ALU_OMOD_2,
                                              GOC_ALU_NEG_A | GOC_ALU_NEG_B | GOC_ALU_NEG_C};

static const uint64_t fmas_capture_values[2][16] = {
    {0x0ULL, 0x80000000ULL, 0x1ULL, 0x80000001ULL, 0x7fffffULL, 0x800000ULL, 0x3f800000ULL,
     0xbf800000ULL, 0x3f800001ULL, 0x3f7fffffULL, 0x40000000ULL, 0x7f7fffffULL, 0x7f800000ULL,
     0xff800000ULL, 0x7f800001ULL, 0xffc00003ULL},
    {0x0ULL, 0x8000000000000000ULL, 0x1ULL, 0x8000000000000001ULL, 0xfffffffffffffULL,
     0x10000000000000ULL, 0x3ff0000000000000ULL, 0xbff0000000000000ULL, 0x3ff0000000000001ULL,
     0x3fefffffffffffffULL, 0x4000000000000000ULL, 0x7fefffffffffffffULL, 0x7ff0000000000000ULL,
     0xfff0000000000000ULL, 0x7ff0000000000001ULL, 0xfff8000000000003ULL},
};

static const uint64_t fmas_capture_digests[2][16] = {
    {0x576afb8106a30db6ULL, 0x17b0667f55864660ULL, 0x7ef3c39e53f15674ULL, 0x52e30545e100e91dULL,
     0x7dfaa75bd9f93fc1ULL, 0x4b82f14f3676d37aULL, 0xde3a498141d5c9aeULL, 0xedccdc2b6bda0332ULL,
     0x9b9aaad4d8a05873ULL, 0xf6008a1d1d0c142cULL, 0x597c7e854253324ULL, 0x2a29c5f3a13a8713ULL,
     0x8ce01722da989723ULL, 0xf3616230e81d97b3ULL, 0x47a892285aa4c257ULL, 0x58fe1e27d8dc4f5bULL},
    {0xcee9c9bf9830281aULL, 0x24838bbedb7284f3ULL, 0x31edd4d575fdd387ULL, 0xc583bfc7054904bbULL,
     0x9c7125ca8c68b45ULL, 0x9aac4ad132b3e7c6ULL, 0xacc029bf67082311ULL, 0x68737b88435eadf7ULL,
     0xea132b28336b725bULL, 0x5587f03df977177bULL, 0xb91e1cf58a3e02afULL, 0x4d0d52e9dcc97ab6ULL,
     0x3810a057a4735093ULL, 0x949a41a1d64a8a1fULL, 0xc12f7090ee8c83a7ULL, 0xbc41179bd09796e6ULL},
};

// Second capture: 4096 xorshift-generated triples with cancellation and tiny
// operands injected into half the cases. The generator below reproduces inputs.
static const uint64_t fmas_random_digests[2][16] = {
    {0xc541e132702ea7b5ULL, 0x4e77491831aea8a8ULL, 0x1c22023b0bda9445ULL, 0x1f90d3062d413b7fULL,
     0xebdc444d802d79d0ULL, 0xcebae882a9c75ee7ULL, 0x38ed35a07d010508ULL, 0x15a41126b93019baULL,
     0x7d7799c6729ae40cULL, 0x78735351bce899a2ULL, 0x810587a7906216faULL, 0x47054733e7051b49ULL,
     0xceb2da620a93078ULL, 0x3a514143a1e51d5fULL, 0xda0ef3cd66be2fcfULL, 0xe18d8b95579cc2d2ULL},
    {0xdd6253dd067567b6ULL, 0x51f48836d04d9123ULL, 0x20348cf7243cb93cULL, 0xd2c8288e073e3e95ULL,
     0x2166916818f8bf23ULL, 0xfe603b8e50cfa26aULL, 0xd69da7e81f920ebULL, 0xf08cf5fa21dd9a6cULL,
     0x9226c35b36ea5dbbULL, 0x23ce8b033259fb07ULL, 0x10784f300ac77478ULL, 0x3ebeb744cbdcd7ddULL,
     0xf3878e292105ed4dULL, 0x9b36a4919d0914b1ULL, 0x2daf81d05b88c0dfULL, 0xcf9ac1db515cec43ULL},
};

// Advance the fixed capture sequence, returning the next raw operand bits.
inline uint32_t fmas_capture_next(uint32_t &state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

// Produce one captured triple for the given format and index. State starts at
// 0x9174ab23 and advances continuously across the 4096 rows of that format.
inline void fmas_capture_random(unsigned op, unsigned index, uint32_t &state, uint64_t (&v)[3]) {
  for (auto &x : v) {
    x = fmas_capture_next(state);
    if (op)
      x |= uint64_t(fmas_capture_next(state)) << 32;
  }
  unsigned fraction = op ? 52 : 23, bias = op ? 1023 : 127;
  if (index % 4 == 0) {
    v[1] = (uint64_t(bias) << fraction) | 1;
    v[2] = v[0] ^ (1ULL << (op ? 63 : 31));
  }
  if (index % 4 == 1) {
    v[0] &= (1ULL << fraction) - 1;
    v[1] = (uint64_t(bias + 1) << fraction) - 1;
    v[2] = 1;
  }
}

// Literal GPU witnesses: fused subnormal rounding (+/-), avoided intermediate
// overflow, OMOD's normal-precision underflow (+/-), and severe cancellation.
// The first two rows yield the smallest subnormal with post-scaling; rounding
// FMA before scaling would incorrectly produce zero.
static const uint64_t fmas_boundary[2][8][19] = {
    {
        {0x2a000001ULL, 0x29ffffffULL, 0x0ULL, 0x14800000ULL, 0x15000000ULL, 0x15800000ULL,
         0x14000000ULL, 0x14800000ULL, 0x94800000ULL, 0x15000000ULL, 0x14800000ULL, 0x1ULL, 0x0ULL,
         0x0ULL, 0x0ULL, 0x1ULL, 0x80000001ULL, 0x0ULL, 0x1ULL},
        {0xaa000001ULL, 0x29ffffffULL, 0x0ULL, 0x94800000ULL, 0x95000000ULL, 0x95800000ULL,
         0x94000000ULL, 0x0ULL, 0x94800000ULL, 0x0ULL, 0x94800000ULL, 0x80000001ULL, 0x0ULL, 0x0ULL,
         0x0ULL, 0x0ULL, 0x80000001ULL, 0x0ULL, 0x80000001ULL},
        {0x7f7fffffULL, 0x40000000ULL, 0x1ULL, 0x7f800000ULL, 0x7f800000ULL, 0x7f800000ULL,
         0x7f800000ULL, 0x3f800000ULL, 0xff800000ULL, 0x3f800000ULL, 0x7f800000ULL, 0x5fffffffULL,
         0x607fffffULL, 0x60ffffffULL, 0x5f7fffffULL, 0x3f800000ULL, 0xdfffffffULL, 0x3f800000ULL,
         0x5fffffffULL},
        {0x800000ULL, 0x3f7fffffULL, 0x0ULL, 0x800000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x800000ULL,
         0x80800000ULL, 0x0ULL, 0x800000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x80000000ULL,
         0x0ULL, 0x0ULL},
        {0x80800000ULL, 0x3f7fffffULL, 0x0ULL, 0x80800000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL,
         0x80800000ULL, 0x0ULL, 0x80800000ULL, 0x80000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL,
         0x80000000ULL, 0x0ULL, 0x80000000ULL},
        {0x1ULL, 0x80000001ULL, 0x800000ULL, 0x800000ULL, 0x1000000ULL, 0x1800000ULL, 0x0ULL,
         0x800000ULL, 0x80800000ULL, 0x1000000ULL, 0x80800000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL,
         0x0ULL, 0x80000000ULL, 0x0ULL, 0x80000000ULL},
        {0x3f800001ULL, 0x3f7fffffULL, 0xbf800000ULL, 0x337ffffeULL, 0x33fffffeULL, 0x347ffffeULL,
         0x32fffffeULL, 0x337ffffeULL, 0xb37ffffeULL, 0x33fffffeULL, 0x40000000ULL, 0x137ffffeULL,
         0x13fffffeULL, 0x147ffffeULL, 0x12fffffeULL, 0x137ffffeULL, 0x937ffffeULL, 0x13fffffeULL,
         0x20000000ULL},
        {0x7f7fffffULL, 0x3f800001ULL, 0xff7fffffULL, 0x73ffffffULL, 0x747fffffULL, 0x74ffffffULL,
         0x737fffffULL, 0x3f800000ULL, 0xf3ffffffULL, 0x3f800000ULL, 0x7f800000ULL, 0x7f800000ULL,
         0x7f800000ULL, 0x7f800000ULL, 0x7f800000ULL, 0x3f800000ULL, 0xff800000ULL, 0x3f800000ULL,
         0x7f800000ULL},
    },
    {
        {0x2260000000000001ULL, 0x224fffffffffffffULL, 0x0ULL, 0x4c0000000000000ULL,
         0x4d0000000000000ULL, 0x4e0000000000000ULL, 0x4b0000000000000ULL, 0x4c0000000000000ULL,
         0x84c0000000000000ULL, 0x4d0000000000000ULL, 0x4c0000000000000ULL, 0x1ULL, 0x0ULL, 0x0ULL,
         0x0ULL, 0x1ULL, 0x8000000000000001ULL, 0x0ULL, 0x1ULL},
        {0xa260000000000001ULL, 0x224fffffffffffffULL, 0x0ULL, 0x84c0000000000000ULL,
         0x84d0000000000000ULL, 0x84e0000000000000ULL, 0x84b0000000000000ULL, 0x0ULL,
         0x84c0000000000000ULL, 0x0ULL, 0x84c0000000000000ULL, 0x8000000000000001ULL, 0x0ULL,
         0x0ULL, 0x0ULL, 0x0ULL, 0x8000000000000001ULL, 0x0ULL, 0x8000000000000001ULL},
        {0x7fefffffffffffffULL, 0x4000000000000000ULL, 0x1ULL, 0x7ff0000000000000ULL,
         0x7ff0000000000000ULL, 0x7ff0000000000000ULL, 0x7ff0000000000000ULL, 0x3ff0000000000000ULL,
         0xfff0000000000000ULL, 0x3ff0000000000000ULL, 0x7ff0000000000000ULL, 0x77ffffffffffffffULL,
         0x780fffffffffffffULL, 0x781fffffffffffffULL, 0x77efffffffffffffULL, 0x3ff0000000000000ULL,
         0xf7ffffffffffffffULL, 0x3ff0000000000000ULL, 0x77ffffffffffffffULL},
        {0x10000000000000ULL, 0x3fefffffffffffffULL, 0x0ULL, 0x10000000000000ULL, 0x0ULL, 0x0ULL,
         0x0ULL, 0x10000000000000ULL, 0x8010000000000000ULL, 0x0ULL, 0x10000000000000ULL, 0x0ULL,
         0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x8000000000000000ULL, 0x0ULL, 0x0ULL},
        {0x8010000000000000ULL, 0x3fefffffffffffffULL, 0x0ULL, 0x8010000000000000ULL, 0x0ULL,
         0x0ULL, 0x0ULL, 0x0ULL, 0x8010000000000000ULL, 0x0ULL, 0x8010000000000000ULL,
         0x8000000000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x8000000000000000ULL, 0x0ULL,
         0x8000000000000000ULL},
        {0x1ULL, 0x8000000000000001ULL, 0x10000000000000ULL, 0x10000000000000ULL,
         0x20000000000000ULL, 0x30000000000000ULL, 0x0ULL, 0x10000000000000ULL,
         0x8010000000000000ULL, 0x20000000000000ULL, 0x8010000000000000ULL, 0x0ULL, 0x0ULL, 0x0ULL,
         0x0ULL, 0x0ULL, 0x8000000000000000ULL, 0x0ULL, 0x8000000000000000ULL},
        {0x3ff0000000000001ULL, 0x3fefffffffffffffULL, 0xbff0000000000000ULL, 0x3c9ffffffffffffeULL,
         0x3caffffffffffffeULL, 0x3cbffffffffffffeULL, 0x3c8ffffffffffffeULL, 0x3c9ffffffffffffeULL,
         0xbc9ffffffffffffeULL, 0x3caffffffffffffeULL, 0x4000000000000000ULL, 0x349ffffffffffffeULL,
         0x34affffffffffffeULL, 0x34bffffffffffffeULL, 0x348ffffffffffffeULL, 0x349ffffffffffffeULL,
         0xb49ffffffffffffeULL, 0x34affffffffffffeULL, 0x3800000000000000ULL},
        {0x7fefffffffffffffULL, 0x3ff0000000000001ULL, 0xffefffffffffffffULL, 0x7cafffffffffffffULL,
         0x7cbfffffffffffffULL, 0x7ccfffffffffffffULL, 0x7c9fffffffffffffULL, 0x3ff0000000000000ULL,
         0xfcafffffffffffffULL, 0x3ff0000000000000ULL, 0x7ff0000000000000ULL, 0x7ff0000000000000ULL,
         0x7ff0000000000000ULL, 0x7ff0000000000000ULL, 0x7ff0000000000000ULL, 0x3ff0000000000000ULL,
         0xfff0000000000000ULL, 0x3ff0000000000000ULL, 0x7ff0000000000000ULL},
    },
};

} // namespace goc_test

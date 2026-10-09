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
    {UINT64_C(0x0), UINT64_C(0x80000000), UINT64_C(0x1), UINT64_C(0x80000001), UINT64_C(0x7fffff),
     UINT64_C(0x800000), UINT64_C(0x3f800000), UINT64_C(0xbf800000), UINT64_C(0x3f800001),
     UINT64_C(0x3f7fffff), UINT64_C(0x40000000), UINT64_C(0x7f7fffff), UINT64_C(0x7f800000),
     UINT64_C(0xff800000), UINT64_C(0x7f800001), UINT64_C(0xffc00003)},
    {UINT64_C(0x0), UINT64_C(0x8000000000000000), UINT64_C(0x1), UINT64_C(0x8000000000000001),
     UINT64_C(0xfffffffffffff), UINT64_C(0x10000000000000), UINT64_C(0x3ff0000000000000),
     UINT64_C(0xbff0000000000000), UINT64_C(0x3ff0000000000001), UINT64_C(0x3fefffffffffffff),
     UINT64_C(0x4000000000000000), UINT64_C(0x7fefffffffffffff), UINT64_C(0x7ff0000000000000),
     UINT64_C(0xfff0000000000000), UINT64_C(0x7ff0000000000001), UINT64_C(0xfff8000000000003)},
};

static const uint64_t fmas_capture_digests[2][16] = {
    {UINT64_C(0x576afb8106a30db6), UINT64_C(0x17b0667f55864660), UINT64_C(0x7ef3c39e53f15674),
     UINT64_C(0x52e30545e100e91d), UINT64_C(0x7dfaa75bd9f93fc1), UINT64_C(0x4b82f14f3676d37a),
     UINT64_C(0xde3a498141d5c9ae), UINT64_C(0xedccdc2b6bda0332), UINT64_C(0x9b9aaad4d8a05873),
     UINT64_C(0xf6008a1d1d0c142c), UINT64_C(0x597c7e854253324), UINT64_C(0x2a29c5f3a13a8713),
     UINT64_C(0x8ce01722da989723), UINT64_C(0xf3616230e81d97b3), UINT64_C(0x47a892285aa4c257),
     UINT64_C(0x58fe1e27d8dc4f5b)},
    {UINT64_C(0xcee9c9bf9830281a), UINT64_C(0x24838bbedb7284f3), UINT64_C(0x31edd4d575fdd387),
     UINT64_C(0xc583bfc7054904bb), UINT64_C(0x9c7125ca8c68b45), UINT64_C(0x9aac4ad132b3e7c6),
     UINT64_C(0xacc029bf67082311), UINT64_C(0x68737b88435eadf7), UINT64_C(0xea132b28336b725b),
     UINT64_C(0x5587f03df977177b), UINT64_C(0xb91e1cf58a3e02af), UINT64_C(0x4d0d52e9dcc97ab6),
     UINT64_C(0x3810a057a4735093), UINT64_C(0x949a41a1d64a8a1f), UINT64_C(0xc12f7090ee8c83a7),
     UINT64_C(0xbc41179bd09796e6)},
};

// Second capture: 4096 xorshift-generated triples with cancellation and tiny
// operands injected into half the cases. The generator below reproduces inputs.
static const uint64_t fmas_random_digests[2][16] = {
    {UINT64_C(0xc541e132702ea7b5), UINT64_C(0x4e77491831aea8a8), UINT64_C(0x1c22023b0bda9445),
     UINT64_C(0x1f90d3062d413b7f), UINT64_C(0xebdc444d802d79d0), UINT64_C(0xcebae882a9c75ee7),
     UINT64_C(0x38ed35a07d010508), UINT64_C(0x15a41126b93019ba), UINT64_C(0x7d7799c6729ae40c),
     UINT64_C(0x78735351bce899a2), UINT64_C(0x810587a7906216fa), UINT64_C(0x47054733e7051b49),
     UINT64_C(0xceb2da620a93078), UINT64_C(0x3a514143a1e51d5f), UINT64_C(0xda0ef3cd66be2fcf),
     UINT64_C(0xe18d8b95579cc2d2)},
    {UINT64_C(0xdd6253dd067567b6), UINT64_C(0x51f48836d04d9123), UINT64_C(0x20348cf7243cb93c),
     UINT64_C(0xd2c8288e073e3e95), UINT64_C(0x2166916818f8bf23), UINT64_C(0xfe603b8e50cfa26a),
     UINT64_C(0xd69da7e81f920eb), UINT64_C(0xf08cf5fa21dd9a6c), UINT64_C(0x9226c35b36ea5dbb),
     UINT64_C(0x23ce8b033259fb07), UINT64_C(0x10784f300ac77478), UINT64_C(0x3ebeb744cbdcd7dd),
     UINT64_C(0xf3878e292105ed4d), UINT64_C(0x9b36a4919d0914b1), UINT64_C(0x2daf81d05b88c0df),
     UINT64_C(0xcf9ac1db515cec43)},
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
    v[2] = v[0] ^ (UINT64_C(1) << (op ? 63 : 31));
  }
  if (index % 4 == 1) {
    v[0] &= (UINT64_C(1) << fraction) - 1;
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
        {UINT64_C(0x2a000001), UINT64_C(0x29ffffff), UINT64_C(0x0), UINT64_C(0x14800000),
         UINT64_C(0x15000000), UINT64_C(0x15800000), UINT64_C(0x14000000), UINT64_C(0x14800000),
         UINT64_C(0x94800000), UINT64_C(0x15000000), UINT64_C(0x14800000), UINT64_C(0x1),
         UINT64_C(0x0), UINT64_C(0x0), UINT64_C(0x0), UINT64_C(0x1), UINT64_C(0x80000001),
         UINT64_C(0x0), UINT64_C(0x1)},
        {UINT64_C(0xaa000001), UINT64_C(0x29ffffff), UINT64_C(0x0), UINT64_C(0x94800000),
         UINT64_C(0x95000000), UINT64_C(0x95800000), UINT64_C(0x94000000), UINT64_C(0x0),
         UINT64_C(0x94800000), UINT64_C(0x0), UINT64_C(0x94800000), UINT64_C(0x80000001),
         UINT64_C(0x0), UINT64_C(0x0), UINT64_C(0x0), UINT64_C(0x0), UINT64_C(0x80000001),
         UINT64_C(0x0), UINT64_C(0x80000001)},
        {UINT64_C(0x7f7fffff), UINT64_C(0x40000000), UINT64_C(0x1), UINT64_C(0x7f800000),
         UINT64_C(0x7f800000), UINT64_C(0x7f800000), UINT64_C(0x7f800000), UINT64_C(0x3f800000),
         UINT64_C(0xff800000), UINT64_C(0x3f800000), UINT64_C(0x7f800000), UINT64_C(0x5fffffff),
         UINT64_C(0x607fffff), UINT64_C(0x60ffffff), UINT64_C(0x5f7fffff), UINT64_C(0x3f800000),
         UINT64_C(0xdfffffff), UINT64_C(0x3f800000), UINT64_C(0x5fffffff)},
        {UINT64_C(0x800000), UINT64_C(0x3f7fffff), UINT64_C(0x0), UINT64_C(0x800000), UINT64_C(0x0),
         UINT64_C(0x0), UINT64_C(0x0), UINT64_C(0x800000), UINT64_C(0x80800000), UINT64_C(0x0),
         UINT64_C(0x800000), UINT64_C(0x0), UINT64_C(0x0), UINT64_C(0x0), UINT64_C(0x0),
         UINT64_C(0x0), UINT64_C(0x80000000), UINT64_C(0x0), UINT64_C(0x0)},
        {UINT64_C(0x80800000), UINT64_C(0x3f7fffff), UINT64_C(0x0), UINT64_C(0x80800000),
         UINT64_C(0x0), UINT64_C(0x0), UINT64_C(0x0), UINT64_C(0x0), UINT64_C(0x80800000),
         UINT64_C(0x0), UINT64_C(0x80800000), UINT64_C(0x80000000), UINT64_C(0x0), UINT64_C(0x0),
         UINT64_C(0x0), UINT64_C(0x0), UINT64_C(0x80000000), UINT64_C(0x0), UINT64_C(0x80000000)},
        {UINT64_C(0x1), UINT64_C(0x80000001), UINT64_C(0x800000), UINT64_C(0x800000),
         UINT64_C(0x1000000), UINT64_C(0x1800000), UINT64_C(0x0), UINT64_C(0x800000),
         UINT64_C(0x80800000), UINT64_C(0x1000000), UINT64_C(0x80800000), UINT64_C(0x0),
         UINT64_C(0x0), UINT64_C(0x0), UINT64_C(0x0), UINT64_C(0x0), UINT64_C(0x80000000),
         UINT64_C(0x0), UINT64_C(0x80000000)},
        {UINT64_C(0x3f800001), UINT64_C(0x3f7fffff), UINT64_C(0xbf800000), UINT64_C(0x337ffffe),
         UINT64_C(0x33fffffe), UINT64_C(0x347ffffe), UINT64_C(0x32fffffe), UINT64_C(0x337ffffe),
         UINT64_C(0xb37ffffe), UINT64_C(0x33fffffe), UINT64_C(0x40000000), UINT64_C(0x137ffffe),
         UINT64_C(0x13fffffe), UINT64_C(0x147ffffe), UINT64_C(0x12fffffe), UINT64_C(0x137ffffe),
         UINT64_C(0x937ffffe), UINT64_C(0x13fffffe), UINT64_C(0x20000000)},
        {UINT64_C(0x7f7fffff), UINT64_C(0x3f800001), UINT64_C(0xff7fffff), UINT64_C(0x73ffffff),
         UINT64_C(0x747fffff), UINT64_C(0x74ffffff), UINT64_C(0x737fffff), UINT64_C(0x3f800000),
         UINT64_C(0xf3ffffff), UINT64_C(0x3f800000), UINT64_C(0x7f800000), UINT64_C(0x7f800000),
         UINT64_C(0x7f800000), UINT64_C(0x7f800000), UINT64_C(0x7f800000), UINT64_C(0x3f800000),
         UINT64_C(0xff800000), UINT64_C(0x3f800000), UINT64_C(0x7f800000)},
    },
    {
        {UINT64_C(0x2260000000000001), UINT64_C(0x224fffffffffffff), UINT64_C(0x0),
         UINT64_C(0x4c0000000000000), UINT64_C(0x4d0000000000000), UINT64_C(0x4e0000000000000),
         UINT64_C(0x4b0000000000000), UINT64_C(0x4c0000000000000), UINT64_C(0x84c0000000000000),
         UINT64_C(0x4d0000000000000), UINT64_C(0x4c0000000000000), UINT64_C(0x1), UINT64_C(0x0),
         UINT64_C(0x0), UINT64_C(0x0), UINT64_C(0x1), UINT64_C(0x8000000000000001), UINT64_C(0x0),
         UINT64_C(0x1)},
        {UINT64_C(0xa260000000000001), UINT64_C(0x224fffffffffffff), UINT64_C(0x0),
         UINT64_C(0x84c0000000000000), UINT64_C(0x84d0000000000000), UINT64_C(0x84e0000000000000),
         UINT64_C(0x84b0000000000000), UINT64_C(0x0), UINT64_C(0x84c0000000000000), UINT64_C(0x0),
         UINT64_C(0x84c0000000000000), UINT64_C(0x8000000000000001), UINT64_C(0x0), UINT64_C(0x0),
         UINT64_C(0x0), UINT64_C(0x0), UINT64_C(0x8000000000000001), UINT64_C(0x0),
         UINT64_C(0x8000000000000001)},
        {UINT64_C(0x7fefffffffffffff), UINT64_C(0x4000000000000000), UINT64_C(0x1),
         UINT64_C(0x7ff0000000000000), UINT64_C(0x7ff0000000000000), UINT64_C(0x7ff0000000000000),
         UINT64_C(0x7ff0000000000000), UINT64_C(0x3ff0000000000000), UINT64_C(0xfff0000000000000),
         UINT64_C(0x3ff0000000000000), UINT64_C(0x7ff0000000000000), UINT64_C(0x77ffffffffffffff),
         UINT64_C(0x780fffffffffffff), UINT64_C(0x781fffffffffffff), UINT64_C(0x77efffffffffffff),
         UINT64_C(0x3ff0000000000000), UINT64_C(0xf7ffffffffffffff), UINT64_C(0x3ff0000000000000),
         UINT64_C(0x77ffffffffffffff)},
        {UINT64_C(0x10000000000000), UINT64_C(0x3fefffffffffffff), UINT64_C(0x0),
         UINT64_C(0x10000000000000), UINT64_C(0x0), UINT64_C(0x0), UINT64_C(0x0),
         UINT64_C(0x10000000000000), UINT64_C(0x8010000000000000), UINT64_C(0x0),
         UINT64_C(0x10000000000000), UINT64_C(0x0), UINT64_C(0x0), UINT64_C(0x0), UINT64_C(0x0),
         UINT64_C(0x0), UINT64_C(0x8000000000000000), UINT64_C(0x0), UINT64_C(0x0)},
        {UINT64_C(0x8010000000000000), UINT64_C(0x3fefffffffffffff), UINT64_C(0x0),
         UINT64_C(0x8010000000000000), UINT64_C(0x0), UINT64_C(0x0), UINT64_C(0x0), UINT64_C(0x0),
         UINT64_C(0x8010000000000000), UINT64_C(0x0), UINT64_C(0x8010000000000000),
         UINT64_C(0x8000000000000000), UINT64_C(0x0), UINT64_C(0x0), UINT64_C(0x0), UINT64_C(0x0),
         UINT64_C(0x8000000000000000), UINT64_C(0x0), UINT64_C(0x8000000000000000)},
        {UINT64_C(0x1), UINT64_C(0x8000000000000001), UINT64_C(0x10000000000000),
         UINT64_C(0x10000000000000), UINT64_C(0x20000000000000), UINT64_C(0x30000000000000),
         UINT64_C(0x0), UINT64_C(0x10000000000000), UINT64_C(0x8010000000000000),
         UINT64_C(0x20000000000000), UINT64_C(0x8010000000000000), UINT64_C(0x0), UINT64_C(0x0),
         UINT64_C(0x0), UINT64_C(0x0), UINT64_C(0x0), UINT64_C(0x8000000000000000), UINT64_C(0x0),
         UINT64_C(0x8000000000000000)},
        {UINT64_C(0x3ff0000000000001), UINT64_C(0x3fefffffffffffff), UINT64_C(0xbff0000000000000),
         UINT64_C(0x3c9ffffffffffffe), UINT64_C(0x3caffffffffffffe), UINT64_C(0x3cbffffffffffffe),
         UINT64_C(0x3c8ffffffffffffe), UINT64_C(0x3c9ffffffffffffe), UINT64_C(0xbc9ffffffffffffe),
         UINT64_C(0x3caffffffffffffe), UINT64_C(0x4000000000000000), UINT64_C(0x349ffffffffffffe),
         UINT64_C(0x34affffffffffffe), UINT64_C(0x34bffffffffffffe), UINT64_C(0x348ffffffffffffe),
         UINT64_C(0x349ffffffffffffe), UINT64_C(0xb49ffffffffffffe), UINT64_C(0x34affffffffffffe),
         UINT64_C(0x3800000000000000)},
        {UINT64_C(0x7fefffffffffffff), UINT64_C(0x3ff0000000000001), UINT64_C(0xffefffffffffffff),
         UINT64_C(0x7cafffffffffffff), UINT64_C(0x7cbfffffffffffff), UINT64_C(0x7ccfffffffffffff),
         UINT64_C(0x7c9fffffffffffff), UINT64_C(0x3ff0000000000000), UINT64_C(0xfcafffffffffffff),
         UINT64_C(0x3ff0000000000000), UINT64_C(0x7ff0000000000000), UINT64_C(0x7ff0000000000000),
         UINT64_C(0x7ff0000000000000), UINT64_C(0x7ff0000000000000), UINT64_C(0x7ff0000000000000),
         UINT64_C(0x3ff0000000000000), UINT64_C(0xfff0000000000000), UINT64_C(0x3ff0000000000000),
         UINT64_C(0x7ff0000000000000)},
    },
};

} // namespace goc_test

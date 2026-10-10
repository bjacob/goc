// SPDX-License-Identifier: MIT
#pragma once

#include <stdint.h>

namespace goc_test {

// RX 9070 gfx1201, RNE. [MIXLO/MIXHI][source low/high+modifiers][guest policy].
// Policy bit 0 flushes inputs; bit 1 flushes outputs, for both FP32 and FP16.
// capture_arithmetic_exceptions.py fma_mixlo/fma_mixhi 16 [--mix-high],
// probe argument 15, 10, 5, 0 respectively. Word-wise FNV over result_lo,
// result_hi, EXCP_FLAG_USER: 2 saturation settings * 16 modifiers * 8192 triples.
static const uint64_t mixed_guest_hashes[2][2][4] = {
    {
        {0x3dcee9f454f146b1ULL, 0x5db49b2b8f9ac478ULL, 0x4f95f967c3c42d5ULL, 0x7aed70ceb2ca0830ULL},
        {0xb69b80155780142cULL, 0x1a2c24f357ae39fbULL, 0x1e4c67bebbe26f2cULL,
         0x5430e92e6b9bdb63ULL},
    },
    {
        {0xdc6f2d02a8dc2325ULL, 0xed8861dd2f032325ULL, 0xfecfea61ef402325ULL,
         0xad61407249bf2325ULL},
        {0x47bb2109440f2325ULL, 0x5041450a6ad02325ULL, 0xf40acb5d1b5f2325ULL,
         0x25fc93ae9cd42325ULL},
    },
};

// MIX_F32, generated with family fma_mix and width 32; same record layout.
static const uint64_t mixed_float_guest_hashes[2][4] = {
    {0x26ab74b826376b29ULL, 0x5056ff74b10a2661ULL, 0x6cac82756180b871ULL, 0xb353dd59bc7e5f0dULL},
    {0x1585b554f35594c5ULL, 0x1dc8b4a594e34665ULL, 0x988fdf70a6a80525ULL, 0x2481051c1e6600a1ULL},
};

} // namespace goc_test

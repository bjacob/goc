// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <stdint.h>

namespace goc_test {

// RX 9070 gfx1201, MODE=0xf0 | (FP16_OVFL << 23), captured 2026-10-08.
// Each format has 4096 triples ordered A outermost, then B, then C. Ten columns
// group by FP16_OVFL, then none, -abs(A)/abs(B)/-C, mul:2, div:2, clamp/mul:4.
// FNV-1a digests consume little-endian result bytes: 4 for FP16/FP32, 8 for FP64.
// FP16 D starts at 0x12345678, and the digests include its preserved upper half.
static const uint32_t fixup_capture_modes[] = {
    0, GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_ABS_B | GOC_ALU_NEG_C, GOC_ALU_OMOD_2,
    GOC_ALU_OMOD_HALF, GOC_ALU_CLAMP | GOC_ALU_OMOD_4};

static const uint64_t fixup_capture_values[3][16] = {
    {0x0ULL, 0x8000ULL, 0x1ULL, 0x8001ULL, 0x3ffULL, 0x400ULL, 0x3c00ULL, 0xbc00ULL, 0x4000ULL,
     0x7bffULL, 0x77ffULL, 0x7c00ULL, 0xfc00ULL, 0x7c01ULL, 0xfe03ULL, 0x800ULL},
    {0x0ULL, 0x80000000ULL, 0x1ULL, 0x80000001ULL, 0x7fffffULL, 0x800000ULL, 0x3f800000ULL,
     0xbf800000ULL, 0x40000000ULL, 0x7f7fffffULL, 0x7effffffULL, 0x7f800000ULL, 0xff800000ULL,
     0x7f800001ULL, 0xffc00003ULL, 0x1000000ULL},
    {0x0ULL, 0x8000000000000000ULL, 0x1ULL, 0x8000000000000001ULL, 0xfffffffffffffULL,
     0x10000000000000ULL, 0x3ff0000000000000ULL, 0xbff0000000000000ULL, 0x4000000000000000ULL,
     0x7fefffffffffffffULL, 0x7fdfffffffffffffULL, 0x7ff0000000000000ULL, 0xfff0000000000000ULL,
     0x7ff0000000000001ULL, 0xfff8000000000003ULL, 0x20000000000000ULL},
};

static const uint64_t fixup_capture_digests[3][10] = {
    {0xb38763c18dba3595ULL, 0x1efd5c5aae38e395ULL, 0xfcc4b255f651f85dULL, 0x97fc9b7b757e47adULL,
     0xeff19f7014208ae5ULL, 0x751a84a64f0a3175ULL, 0x60f518ad4b5fa75ULL, 0x84c5a39f06818835ULL,
     0xf50955c0725c1edULL, 0xeff19f7014208ae5ULL},
    {0x8f2fd19a5eb283b1ULL, 0x9de6b4e11082acb1ULL, 0x7519dbec75609621ULL, 0x5a2a1c7e9c4616f1ULL,
     0x693ff367df6ff8c5ULL, 0x8f2fd19a5eb283b1ULL, 0x9de6b4e11082acb1ULL, 0x7519dbec75609621ULL,
     0x5a2a1c7e9c4616f1ULL, 0x693ff367df6ff8c5ULL},
    {0x266ed547042c4769ULL, 0x6e00ab1dc302a869ULL, 0x4b7ae2cff5842161ULL, 0xb68d3c7010cf62c9ULL,
     0x49c775e21a5725a5ULL, 0x266ed547042c4769ULL, 0x6e00ab1dc302a869ULL, 0x4b7ae2cff5842161ULL,
     0xb68d3c7010cf62c9ULL, 0x49c775e21a5725a5ULL},
};

} // namespace goc_test

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
    {UINT64_C(0x0), UINT64_C(0x8000), UINT64_C(0x1), UINT64_C(0x8001), UINT64_C(0x3ff),
     UINT64_C(0x400), UINT64_C(0x3c00), UINT64_C(0xbc00), UINT64_C(0x4000), UINT64_C(0x7bff),
     UINT64_C(0x77ff), UINT64_C(0x7c00), UINT64_C(0xfc00), UINT64_C(0x7c01), UINT64_C(0xfe03),
     UINT64_C(0x800)},
    {UINT64_C(0x0), UINT64_C(0x80000000), UINT64_C(0x1), UINT64_C(0x80000001), UINT64_C(0x7fffff),
     UINT64_C(0x800000), UINT64_C(0x3f800000), UINT64_C(0xbf800000), UINT64_C(0x40000000),
     UINT64_C(0x7f7fffff), UINT64_C(0x7effffff), UINT64_C(0x7f800000), UINT64_C(0xff800000),
     UINT64_C(0x7f800001), UINT64_C(0xffc00003), UINT64_C(0x1000000)},
    {UINT64_C(0x0), UINT64_C(0x8000000000000000), UINT64_C(0x1), UINT64_C(0x8000000000000001),
     UINT64_C(0xfffffffffffff), UINT64_C(0x10000000000000), UINT64_C(0x3ff0000000000000),
     UINT64_C(0xbff0000000000000), UINT64_C(0x4000000000000000), UINT64_C(0x7fefffffffffffff),
     UINT64_C(0x7fdfffffffffffff), UINT64_C(0x7ff0000000000000), UINT64_C(0xfff0000000000000),
     UINT64_C(0x7ff0000000000001), UINT64_C(0xfff8000000000003), UINT64_C(0x20000000000000)},
};

static const uint64_t fixup_capture_digests[3][10] = {
    {UINT64_C(0xb38763c18dba3595), UINT64_C(0x1efd5c5aae38e395), UINT64_C(0xfcc4b255f651f85d),
     UINT64_C(0x97fc9b7b757e47ad), UINT64_C(0xeff19f7014208ae5), UINT64_C(0x751a84a64f0a3175),
     UINT64_C(0x60f518ad4b5fa75), UINT64_C(0x84c5a39f06818835), UINT64_C(0xf50955c0725c1ed),
     UINT64_C(0xeff19f7014208ae5)},
    {UINT64_C(0x8f2fd19a5eb283b1), UINT64_C(0x9de6b4e11082acb1), UINT64_C(0x7519dbec75609621),
     UINT64_C(0x5a2a1c7e9c4616f1), UINT64_C(0x693ff367df6ff8c5), UINT64_C(0x8f2fd19a5eb283b1),
     UINT64_C(0x9de6b4e11082acb1), UINT64_C(0x7519dbec75609621), UINT64_C(0x5a2a1c7e9c4616f1),
     UINT64_C(0x693ff367df6ff8c5)},
    {UINT64_C(0x266ed547042c4769), UINT64_C(0x6e00ab1dc302a869), UINT64_C(0x4b7ae2cff5842161),
     UINT64_C(0xb68d3c7010cf62c9), UINT64_C(0x49c775e21a5725a5), UINT64_C(0x266ed547042c4769),
     UINT64_C(0x6e00ab1dc302a869), UINT64_C(0x4b7ae2cff5842161), UINT64_C(0xb68d3c7010cf62c9),
     UINT64_C(0x49c775e21a5725a5)},
};

} // namespace goc_test

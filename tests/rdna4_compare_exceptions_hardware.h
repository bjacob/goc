// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc_test {

// RX 9070 / gfx1201 capture: 112 scalar/CMP/CMPX predicates, four denormal
// modes, full/zero/single-lane EXEC, 12x12 operand pairs (193,536 register reads).
// An additional 145,152 vector cases cover signaling (CLAMP) comparisons.
// Tables are indexed by input-denorm-preserved + 2*signaling. Scalar instructions ignore
// EXEC; vector instructions with empty EXEC generate no flags. Output denormal
// mode has no effect. Rows select A, columns select B from compare_exception_inputs.
static const uint8_t compare_exception_hardware[4][144] = {
    {
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
    },
    {
        0, 0, 2, 2, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 2, 2, 0, 0, 0, 0, 1, 0, 0, 2, 2, 2, 2, 2, 2,
        2, 0, 0, 1, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 1, 0, 0, 2, 0, 0, 2, 2, 0, 0, 0, 0, 1, 0,
        0, 2, 0, 0, 2, 2, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 1, 0, 0, 2,
    },
    {
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
    },
    {
        0, 0, 2, 2, 0, 0, 0, 0, 1, 1, 1, 2, 0, 0, 2, 2, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2,
        2, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0, 0, 1, 1, 1, 2, 0, 0, 2, 2, 0, 0, 0, 0, 1, 1,
        1, 2, 0, 0, 2, 2, 0, 0, 0, 0, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 1, 1, 1, 2,
    },

};

static const uint64_t compare_exception_inputs[3][12] = {
    {0x0ULL, 0x8000ULL, 0x1ULL, 0x3ffULL, 0x400ULL, 0x3c00ULL, 0x7c00ULL, 0xfc00ULL, 0x7c01ULL,
     0x7e00ULL, 0xfe01ULL, 0x8001ULL},
    {0x0ULL, 0x80000000ULL, 0x1ULL, 0x7fffffULL, 0x800000ULL, 0x3f800000ULL, 0x7f800000ULL,
     0xff800000ULL, 0x7f800001ULL, 0x7fc00000ULL, 0xffc00001ULL, 0x80000001ULL},
    {0x0ULL, 0x8000000000000000ULL, 0x1ULL, 0xfffffffffffffULL, 0x10000000000000ULL,
     0x3ff0000000000000ULL, 0x7ff0000000000000ULL, 0xfff0000000000000ULL, 0x7ff0000000000001ULL,
     0x7ff8000000000000ULL, 0xfff8000000000001ULL, 0x8000000000000001ULL},
};

} // namespace goc_test

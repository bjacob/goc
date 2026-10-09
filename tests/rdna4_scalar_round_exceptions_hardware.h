// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc_test {

// RX 9070 / gfx1201: all 65,536 FP16 encodings and 65,536 FP32 patterns with
// every sign/exponent/high-mantissa combination and varied low bits. All eight
// scalar rounding instructions, four denormal modes and three EXEC masks agree
// (6,291,456 flag reads). Columns: FP16, FP32. Rows: flush/preserve input denorms.
static const uint64_t scalar_round_exception_hashes[2][2] = {
    {0x36675f6029342325ULL, 0xd261d664c585b3a5ULL},
    {0xb2f82b3815a86b25ULL, 0xf3df6de68bd25fa5ULL},
};

} // namespace goc_test

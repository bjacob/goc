// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc_test {

// RX 9070 gfx1201, RNE; [FMAMK/FMAAK][flush input/output policy bits].
// capture_arithmetic_exceptions.py fmamk/fmaak 32; probe MODE nibble 15/10/5/0.
// Word-wise FNV over the first 8192 result_lo records. Literal is 0x3f800001.
// Exception records are intentionally excluded: reporting remains unsupported.
static const uint64_t literal_fma32_hashes[2][4] = {
    {0xcdb655d19bd382a0ULL, 0x54893c94dd346317ULL, 0xb0fc9036f41f7120ULL, 0x54893c94dd346317ULL},
    {0x2b75b8c12f88ebc9ULL, 0x46c7591f105a9866ULL, 0x2b75b8c12f88ebc9ULL, 0x46c7591f105a9866ULL},
};

} // namespace goc_test

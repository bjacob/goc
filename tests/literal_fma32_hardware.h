// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc_test {

// RX 9070 gfx1201, RNE; [FMAMK/FMAAK][flush input/output policy bits].
// capture_arithmetic_exceptions.py fmamk/fmaak 32; probe MODE nibble 15/10/5/0.
// Word-wise FNV over the first 8192 result_lo records. Literal is 0x3f800001.
// Result records only; exception records are checked separately below.
static const uint64_t literal_fma32_hashes[2][4] = {
    {0xcdb655d19bd382a0ULL, 0x54893c94dd346317ULL, 0xb0fc9036f41f7120ULL, 0x54893c94dd346317ULL},
    {0x2b75b8c12f88ebc9ULL, 0x46c7591f105a9866ULL, 0x2b75b8c12f88ebc9ULL, 0x46c7591f105a9866ULL},
};

// Same corpus, word-wise FNV over each exception record.
static const uint64_t literal_fma32_exception_hashes[2][4] = {
    {0x53c5bd4fcf3b6b84ULL, 0x2900de66bdb0b14cULL, 0x73222dbc3315eb84ULL, 0x2900de66bdb0b14cULL},
    {0x485490db790b7cecULL, 0x7902c2538eef938ULL, 0x485490db790b7cecULL, 0x7902c2538eef938ULL},
};

// --rounding-boundaries --literal 0x3f7fffff / 0x3f800001; result then flags.
// [FMAMK/FMAAK][literal][flush policy], first 8192 records, RX 9070 gfx1201.
static const uint64_t literal_fma32_boundary_hashes[2][2][4] = {
    {
        {0xb4adf72c101177e5ULL, 0xb1d70613a8cc2ca5ULL, 0x7fdba16d51304bb5ULL,
         0x2579f4734ecc2ca5ULL},
        {0xda686f675d0227fdULL, 0x9940ed17c67fa1c5ULL, 0x59b6de84671a5d6dULL,
         0x9940ed17c67fa1c5ULL},
    },
    {
        {0xad1b17a0cafa2fc5ULL, 0x126f3a9a843abfc5ULL, 0xad1b17a0cafa2fc5ULL,
         0x126f3a9a843abfc5ULL},
        {0x3340e439b697afc5ULL, 0x93bff68fe8103fc5ULL, 0x3340e439b697afc5ULL,
         0x93bff68fe8103fc5ULL},
    },
};

} // namespace goc_test

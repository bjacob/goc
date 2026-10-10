// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc_test {

// RX 9070 gfx1201 RNE. capture_arithmetic_exceptions.py fma/fmac 32,
// with/without --rounding-boundaries; MODE denorm nibble 15/10/5/0.
// Word-wise FNV over result_lo for the first 16 * 8192 records, no FP16_OVFL.
// [FMA/FMAC][edge-random/boundary][flush input/output policy].
static const uint64_t fma32_exact_hashes[2][2][4] = {
    {
        {0xe44ecc04e2368a6cULL, 0x72abb0c144e506d7ULL, 0x3ef18bcfa9b7dac1ULL,
         0x72abb0c144e506d7ULL},
        {0x6db662f2d4a6b1f1ULL, 0xc096156749bb471dULL, 0xdd56294fdc7b1491ULL,
         0x3e9701d5d1f535b5ULL},
    },
    {
        {0x9ff175878018849dULL, 0x7fd82a6ec0151d9ULL, 0x6b059766ec2603e5ULL, 0x63a468791361bfd9ULL},
        {0x25251512f2303819ULL, 0x86dd64be15ccfc1dULL, 0xa7d72d757be49629ULL,
         0x8945fb35bf0e0f35ULL},
    },
};

// Same records and ordering, word-wise FNV over exception flags.
static const uint64_t fma32_exception_hashes[2][2][4] = {
    {
        {0x3e1d3009a948f9f5ULL, 0x4ad51e0d99e86765ULL, 0x610168abb03b6835ULL,
         0x4ad51e0d99e86765ULL},
        {0xfae300bed3828925ULL, 0x2329fa64596318a5ULL, 0x916e9abf6a0f7e25ULL,
         0x7609f4d8a5cf87a5ULL},
    },
    {
        {0xdeb526185fb66f05ULL, 0xab8abd78c789ff85ULL, 0xda1b2656656ba885ULL,
         0xab8abd78c789ff85ULL},
        {0x18be22aa6f4136e5ULL, 0x9439bbf3ba6b0fa5ULL, 0xddc92cfc58f0ec85ULL,
         0xe719b66806d77ea5ULL},
    },
};

} // namespace goc_test

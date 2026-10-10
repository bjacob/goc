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

} // namespace goc_test

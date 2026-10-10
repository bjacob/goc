// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc_test {

// RX 9070 gfx1201 RNE, capture_arithmetic_exceptions.py --vopd [--vopd-y].
// Both VOPD slots agree and match ordinary results and exception flags.
// Word-wise FNV over first 8192 (result_lo, flags) records, literal 0x3f800001.
// [FMAC/FMAMK/FMAAK][input/output flush policy], MODE nibble 15/10/5/0.
static const uint64_t dual_fma_hashes[3][4] = {
    {0x6fe51a083de510e6ULL, 0xcc39d4042dd8d755ULL, 0xbeaa2f595f8397b5ULL, 0xcc39d4042dd8d755ULL},
    {0x12768fa1511b2387ULL, 0x730445c4d6e17f02ULL, 0x4227c95c08f9d827ULL, 0x730445c4d6e17f02ULL},
    {0x141a1f59315f4cb6ULL, 0x711d075055c2f1fbULL, 0x141a1f59315f4cb6ULL, 0x711d075055c2f1fbULL},
};

} // namespace goc_test

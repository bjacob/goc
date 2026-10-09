// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

// GFX1201 / HIP 7.13; scalar_convert_inputs, 65536 raw results per digest.
// Every FP16 pattern in each selected half; integer/FP32 boundary/random inputs.
// Full/empty/alternating EXEC agree; SCC preserved. No NaN normalization.
// Indexed by scalar_fp_flags state, then scalar_convert_call operation.
namespace goc_test {

inline const uint64_t scalar_convert_hardware[8][8] = {
    {
        0x3a8ab8e52608b567ULL,
        0xccd667aa805a8afeULL,
        0xc1d731d75c02f97aULL,
        0x37c8392ca1a8f3bdULL,
        0xa396f0f2d61d24e9ULL,
        0x7505ef0fe4422325ULL,
        0x7505ef0fe4422325ULL,
        0x8e75703b1a37054dULL,
    },
    {
        0x3a8ab8e52608b567ULL,
        0xccd667aa805a8afeULL,
        0xc1d731d75c02f97aULL,
        0x37c8392ca1a8f3bdULL,
        0xa396f0f2d61d24e9ULL,
        0x99a56dea6d882325ULL,
        0x99a56dea6d882325ULL,
        0x8e75703b1a37054dULL,
    },
    {
        0x3a8ab8e52608b567ULL,
        0xccd667aa805a8afeULL,
        0xc1d731d75c02f97aULL,
        0x37c8392ca1a8f3bdULL,
        0x1d29e0bc1e83c06dULL,
        0x7505ef0fe4422325ULL,
        0x7505ef0fe4422325ULL,
        0xce2d026aed063d08ULL,
    },
    {
        0x3a8ab8e52608b567ULL,
        0xccd667aa805a8afeULL,
        0xc1d731d75c02f97aULL,
        0x37c8392ca1a8f3bdULL,
        0x1d29e0bc1e83c06dULL,
        0x99a56dea6d882325ULL,
        0x99a56dea6d882325ULL,
        0xce2d026aed063d08ULL,
    },
    {
        0x3a8ab8e52608b567ULL,
        0xccd667aa805a8afeULL,
        0xc1d731d75c02f97aULL,
        0x37c8392ca1a8f3bdULL,
        0x05902a9e860d8e94ULL,
        0x7505ef0fe4422325ULL,
        0x7505ef0fe4422325ULL,
        0x8e75703b1a37054dULL,
    },
    {
        0x3a8ab8e52608b567ULL,
        0xccd667aa805a8afeULL,
        0xc1d731d75c02f97aULL,
        0x37c8392ca1a8f3bdULL,
        0x05902a9e860d8e94ULL,
        0x99a56dea6d882325ULL,
        0x99a56dea6d882325ULL,
        0x8e75703b1a37054dULL,
    },
    {
        0x3a8ab8e52608b567ULL,
        0xccd667aa805a8afeULL,
        0xc1d731d75c02f97aULL,
        0x37c8392ca1a8f3bdULL,
        0xb25083f4acda1a90ULL,
        0x7505ef0fe4422325ULL,
        0x7505ef0fe4422325ULL,
        0xce2d026aed063d08ULL,
    },
    {
        0x3a8ab8e52608b567ULL,
        0xccd667aa805a8afeULL,
        0xc1d731d75c02f97aULL,
        0x37c8392ca1a8f3bdULL,
        0xb25083f4acda1a90ULL,
        0x99a56dea6d882325ULL,
        0x99a56dea6d882325ULL,
        0xce2d026aed063d08ULL,
    },
};

} // namespace goc_test

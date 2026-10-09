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
        UINT64_C(0x3a8ab8e52608b567),
        UINT64_C(0xccd667aa805a8afe),
        UINT64_C(0xc1d731d75c02f97a),
        UINT64_C(0x37c8392ca1a8f3bd),
        UINT64_C(0xa396f0f2d61d24e9),
        UINT64_C(0x7505ef0fe4422325),
        UINT64_C(0x7505ef0fe4422325),
        UINT64_C(0x8e75703b1a37054d),
    },
    {
        UINT64_C(0x3a8ab8e52608b567),
        UINT64_C(0xccd667aa805a8afe),
        UINT64_C(0xc1d731d75c02f97a),
        UINT64_C(0x37c8392ca1a8f3bd),
        UINT64_C(0xa396f0f2d61d24e9),
        UINT64_C(0x99a56dea6d882325),
        UINT64_C(0x99a56dea6d882325),
        UINT64_C(0x8e75703b1a37054d),
    },
    {
        UINT64_C(0x3a8ab8e52608b567),
        UINT64_C(0xccd667aa805a8afe),
        UINT64_C(0xc1d731d75c02f97a),
        UINT64_C(0x37c8392ca1a8f3bd),
        UINT64_C(0x1d29e0bc1e83c06d),
        UINT64_C(0x7505ef0fe4422325),
        UINT64_C(0x7505ef0fe4422325),
        UINT64_C(0xce2d026aed063d08),
    },
    {
        UINT64_C(0x3a8ab8e52608b567),
        UINT64_C(0xccd667aa805a8afe),
        UINT64_C(0xc1d731d75c02f97a),
        UINT64_C(0x37c8392ca1a8f3bd),
        UINT64_C(0x1d29e0bc1e83c06d),
        UINT64_C(0x99a56dea6d882325),
        UINT64_C(0x99a56dea6d882325),
        UINT64_C(0xce2d026aed063d08),
    },
    {
        UINT64_C(0x3a8ab8e52608b567),
        UINT64_C(0xccd667aa805a8afe),
        UINT64_C(0xc1d731d75c02f97a),
        UINT64_C(0x37c8392ca1a8f3bd),
        UINT64_C(0x05902a9e860d8e94),
        UINT64_C(0x7505ef0fe4422325),
        UINT64_C(0x7505ef0fe4422325),
        UINT64_C(0x8e75703b1a37054d),
    },
    {
        UINT64_C(0x3a8ab8e52608b567),
        UINT64_C(0xccd667aa805a8afe),
        UINT64_C(0xc1d731d75c02f97a),
        UINT64_C(0x37c8392ca1a8f3bd),
        UINT64_C(0x05902a9e860d8e94),
        UINT64_C(0x99a56dea6d882325),
        UINT64_C(0x99a56dea6d882325),
        UINT64_C(0x8e75703b1a37054d),
    },
    {
        UINT64_C(0x3a8ab8e52608b567),
        UINT64_C(0xccd667aa805a8afe),
        UINT64_C(0xc1d731d75c02f97a),
        UINT64_C(0x37c8392ca1a8f3bd),
        UINT64_C(0xb25083f4acda1a90),
        UINT64_C(0x7505ef0fe4422325),
        UINT64_C(0x7505ef0fe4422325),
        UINT64_C(0xce2d026aed063d08),
    },
    {
        UINT64_C(0x3a8ab8e52608b567),
        UINT64_C(0xccd667aa805a8afe),
        UINT64_C(0xc1d731d75c02f97a),
        UINT64_C(0x37c8392ca1a8f3bd),
        UINT64_C(0xb25083f4acda1a90),
        UINT64_C(0x99a56dea6d882325),
        UINT64_C(0x99a56dea6d882325),
        UINT64_C(0xce2d026aed063d08),
    },
};

} // namespace goc_test

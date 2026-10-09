// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

// GFX1201 / HIP 7.13; scalar_round_inputs, 65536 raw results per digest.
// Every FP16 pattern; FP32 boundaries and deterministic random bits. No NaN
// normalization. SCC preserved; full/empty/alternating EXEC captures agree.
// Indexed by scalar_fp_flags state, then scalar_round_functions instruction.
namespace goc_test {

inline const uint64_t scalar_round_hardware[8][8] = {
    {
        0x5e35154a20544af8ULL,
        0x32aeac39129c8b25ULL,
        0x6ce2c88dfdf74f8cULL,
        0x87b7a6fa6e0c3325ULL,
        0xaec273a8d443dd50ULL,
        0x7eeca61247d20b25ULL,
        0x93dd203a4ad4c934ULL,
        0xb7902f3f078dab25ULL,
    },
    {
        0xe0853e3eab544af8ULL,
        0x10a3730ec33c2725ULL,
        0x0f4da91456774f8cULL,
        0x57d119fe0cb40725ULL,
        0xaec273a8d443dd50ULL,
        0x7eeca61247d20b25ULL,
        0x93dd203a4ad4c934ULL,
        0xb7902f3f078dab25ULL,
    },
    {
        0x5e35154a20544af8ULL,
        0x32aeac39129c8b25ULL,
        0x6ce2c88dfdf74f8cULL,
        0x87b7a6fa6e0c3325ULL,
        0xaec273a8d443dd50ULL,
        0x7eeca61247d20b25ULL,
        0x93dd203a4ad4c934ULL,
        0xb7902f3f078dab25ULL,
    },
    {
        0xe0853e3eab544af8ULL,
        0x10a3730ec33c2725ULL,
        0x0f4da91456774f8cULL,
        0x57d119fe0cb40725ULL,
        0xaec273a8d443dd50ULL,
        0x7eeca61247d20b25ULL,
        0x93dd203a4ad4c934ULL,
        0xb7902f3f078dab25ULL,
    },
    {
        0x5e35154a20544af8ULL,
        0x32aeac39129c8b25ULL,
        0x6ce2c88dfdf74f8cULL,
        0x87b7a6fa6e0c3325ULL,
        0xaec273a8d443dd50ULL,
        0x7eeca61247d20b25ULL,
        0x93dd203a4ad4c934ULL,
        0xb7902f3f078dab25ULL,
    },
    {
        0xe0853e3eab544af8ULL,
        0x10a3730ec33c2725ULL,
        0x0f4da91456774f8cULL,
        0x57d119fe0cb40725ULL,
        0xaec273a8d443dd50ULL,
        0x7eeca61247d20b25ULL,
        0x93dd203a4ad4c934ULL,
        0xb7902f3f078dab25ULL,
    },
    {
        0x5e35154a20544af8ULL,
        0x32aeac39129c8b25ULL,
        0x6ce2c88dfdf74f8cULL,
        0x87b7a6fa6e0c3325ULL,
        0xaec273a8d443dd50ULL,
        0x7eeca61247d20b25ULL,
        0x93dd203a4ad4c934ULL,
        0xb7902f3f078dab25ULL,
    },
    {
        0xe0853e3eab544af8ULL,
        0x10a3730ec33c2725ULL,
        0x0f4da91456774f8cULL,
        0x57d119fe0cb40725ULL,
        0xaec273a8d443dd50ULL,
        0x7eeca61247d20b25ULL,
        0x93dd203a4ad4c934ULL,
        0xb7902f3f078dab25ULL,
    },
};

} // namespace goc_test

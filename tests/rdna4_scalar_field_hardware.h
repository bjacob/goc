// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

// GFX1201 / HIP 7.13; scalar_field_inputs, 16384 pairs per instruction.
// Digests fold result low/high and SCC. Full, empty, alternating EXEC agree.
// Indexed by incoming SCC and scalar_field_names order.
namespace goc_test {

inline const uint64_t scalar_field_hardware[2][20] = {
    {
        0x4327da5b468c55c7ULL, 0xc6fd8609f09d1633ULL, 0xb6547c4623886818ULL, 0x68e483d023ebbcbdULL,
        0x6dcb36900f8269a5ULL, 0xbe070fb5ea1c2065ULL, 0x1ccbf92291cd9babULL, 0x7a557680b44e81e3ULL,
        0xe77f0ccf9e960947ULL, 0xb02bdc458955b89bULL, 0x14cae6a19de0d7e2ULL, 0xd6c493779e5975e2ULL,
        0xfbb166544eac73e3ULL, 0x46d104077f7141bfULL, 0xcbb8849c37fd5aebULL, 0xad6f3507b3e1d7aeULL,
        0x85b161f3fc6af5a5ULL, 0x8e9e2131da85eae5ULL, 0x725e4fa64d77c1a5ULL, 0x51c80aabada3fce5ULL,
    },
    {
        0x4327da5b468c55c7ULL, 0xc6fd8609f09d1633ULL, 0xb6547c4623886818ULL, 0x68e483d023ebbcbdULL,
        0x79d6c545eca585a5ULL, 0x876f38dd61238965ULL, 0x1ccbf92291cd9babULL, 0x7a557680b44e81e3ULL,
        0xe77f0ccf9e960947ULL, 0xb02bdc458955b89bULL, 0x27dbea24ca048996ULL, 0x734bb5c135aac396ULL,
        0x3a2f9ea6795f7913ULL, 0x59b141b90088bd23ULL, 0xef565087e438d087ULL, 0x0453c8885076d4b2ULL,
        0x0d1ed05d639993a5ULL, 0xdb8e3629b1d75ce5ULL, 0x02c2da38fb065da5ULL, 0x6162307d399c5be5ULL,
    },
};

} // namespace goc_test

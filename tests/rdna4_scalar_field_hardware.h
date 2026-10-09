// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

// GFX1201 / HIP 7.13; scalar_field_inputs, 16384 pairs per instruction.
// Digests fold result low/high and SCC. Full, empty, alternating EXEC agree.
// Indexed by incoming SCC and scalar_field_names order.
namespace goc_test {

inline const uint64_t scalar_field_hardware[2][20] = {
    {
        UINT64_C(0x4327da5b468c55c7), UINT64_C(0xc6fd8609f09d1633), UINT64_C(0xb6547c4623886818),
        UINT64_C(0x68e483d023ebbcbd), UINT64_C(0x6dcb36900f8269a5), UINT64_C(0xbe070fb5ea1c2065),
        UINT64_C(0x1ccbf92291cd9bab), UINT64_C(0x7a557680b44e81e3), UINT64_C(0xe77f0ccf9e960947),
        UINT64_C(0xb02bdc458955b89b), UINT64_C(0x14cae6a19de0d7e2), UINT64_C(0xd6c493779e5975e2),
        UINT64_C(0xfbb166544eac73e3), UINT64_C(0x46d104077f7141bf), UINT64_C(0xcbb8849c37fd5aeb),
        UINT64_C(0xad6f3507b3e1d7ae), UINT64_C(0x85b161f3fc6af5a5), UINT64_C(0x8e9e2131da85eae5),
        UINT64_C(0x725e4fa64d77c1a5), UINT64_C(0x51c80aabada3fce5),
    },
    {
        UINT64_C(0x4327da5b468c55c7), UINT64_C(0xc6fd8609f09d1633), UINT64_C(0xb6547c4623886818),
        UINT64_C(0x68e483d023ebbcbd), UINT64_C(0x79d6c545eca585a5), UINT64_C(0x876f38dd61238965),
        UINT64_C(0x1ccbf92291cd9bab), UINT64_C(0x7a557680b44e81e3), UINT64_C(0xe77f0ccf9e960947),
        UINT64_C(0xb02bdc458955b89b), UINT64_C(0x27dbea24ca048996), UINT64_C(0x734bb5c135aac396),
        UINT64_C(0x3a2f9ea6795f7913), UINT64_C(0x59b141b90088bd23), UINT64_C(0xef565087e438d087),
        UINT64_C(0x0453c8885076d4b2), UINT64_C(0x0d1ed05d639993a5), UINT64_C(0xdb8e3629b1d75ce5),
        UINT64_C(0x02c2da38fb065da5), UINT64_C(0x6162307d399c5be5),
    },
};

} // namespace goc_test

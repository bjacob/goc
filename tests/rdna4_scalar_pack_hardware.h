// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

// GFX1201 / HIP 7.13; scalar_integer_inputs, 4096 result low/high/SCC triples.
// Full/empty/alternating EXEC agree. Indexed by incoming SCC then opcode.
namespace goc_test {

inline const uint64_t scalar_pack_hardware[2][11] = {
    {
        UINT64_C(0x6ba44db1d6e52625),
        UINT64_C(0x5977be2aa1912625),
        UINT64_C(0x7c9c35510e2ca677),
        UINT64_C(0xf7c8718a09b6a677),
        UINT64_C(0x5d79d49528d79e2b),
        UINT64_C(0x87f89cfab9fd1695),
        UINT64_C(0xcc4ba1dbd79cefc5),
        UINT64_C(0xefd9cf5ba0d5fc75),
        UINT64_C(0x09ac3884172e4075),
        UINT64_C(0x172ca8d38060cd65),
        UINT64_C(0xdb99457bc166bca5),
    },
    {
        UINT64_C(0xb736143ccbb27f25),
        UINT64_C(0xeac0cb0143027f25),
        UINT64_C(0x3f9ab57dd1501ea3),
        UINT64_C(0xa86aa921d75e1ea3),
        UINT64_C(0xf2e7d0cacc30984b),
        UINT64_C(0x9a391ab4a76a7f25),
        UINT64_C(0x35e43f2f86429125),
        UINT64_C(0xefd9cf5ba0d5fc75),
        UINT64_C(0x09ac3884172e4075),
        UINT64_C(0x172ca8d38060cd65),
        UINT64_C(0xdb99457bc166bca5),
    },
};

} // namespace goc_test

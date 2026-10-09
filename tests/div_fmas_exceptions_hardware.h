// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc_test {

// RX 9070 gfx1201, MODE 0xf0. 4,096 edge triples and 4,096 random triples.
// Rows: FP32/64, then condition false/true. Columns: OMOD, CLAMP, source modifiers.
// Regenerate with capture_arithmetic_exceptions.py div_fmas 32/64.
static const uint64_t div_fmas_exception_hashes[2][2][16] = {
    {
        {
            0x182125d98f8f1c65ULL,
            0x6d57221b65fd4095ULL,
            0x9cacba7392706795ULL,
            0x63fe985e81f87915ULL,
            0xb9d103fd6854a325ULL,
            0xb9d103fd6854a325ULL,
            0xb9d103fd6854a325ULL,
            0xb9d103fd6854a325ULL,
            0xb20183a263a3e1a5ULL,
            0x73deaef72d6146b5ULL,
            0xd74685d9c2081ad5ULL,
            0xd5cad258d57832d5ULL,
            0xb9d103fd6854a325ULL,
            0xb9d103fd6854a325ULL,
            0xb9d103fd6854a325ULL,
            0xb9d103fd6854a325ULL,
        },
        {
            0x30e9de2d4126236dULL,
            0x40b35480ea38eabdULL,
            0x79217253e5364d85ULL,
            0x415646937b268b1dULL,
            0xb9d103fd6854a325ULL,
            0xb9d103fd6854a325ULL,
            0xb9d103fd6854a325ULL,
            0xb9d103fd6854a325ULL,
            0xf006aa85195fb74dULL,
            0xa2e33232fc33bdf5ULL,
            0x33ff1311975db4edULL,
            0x1182d75180f80dfdULL,
            0xb9d103fd6854a325ULL,
            0xb9d103fd6854a325ULL,
            0xb9d103fd6854a325ULL,
            0xb9d103fd6854a325ULL,
        },
    },
    {
        {
            0x04c06db180415f2dULL,
            0x67f41df373593b35ULL,
            0x1d2c8df890cd0475ULL,
            0x5e61aecda0428ffdULL,
            0xb9d103fd6854a325ULL,
            0xb9d103fd6854a325ULL,
            0xb9d103fd6854a325ULL,
            0xb9d103fd6854a325ULL,
            0x9ea0cb7a5456246dULL,
            0x6e7baacf3abd4155ULL,
            0x57c6595ec064b7b5ULL,
            0xd02de8c7f3c249bdULL,
            0xb9d103fd6854a325ULL,
            0xb9d103fd6854a325ULL,
            0xb9d103fd6854a325ULL,
            0xb9d103fd6854a325ULL,
        },
        {
            0xbe412d4041e6df4dULL,
            0x91d3d519373c6ca5ULL,
            0x2dd547b060029d15ULL,
            0x3adb0b6940c8ec8dULL,
            0xb9d103fd6854a325ULL,
            0xb9d103fd6854a325ULL,
            0xb9d103fd6854a325ULL,
            0xb9d103fd6854a325ULL,
            0xc9bce1357007016dULL,
            0x620065d73d0def85ULL,
            0xfe01d86e65d41ff5ULL,
            0x0b079c27469a6f6dULL,
            0xb9d103fd6854a325ULL,
            0xb9d103fd6854a325ULL,
            0xb9d103fd6854a325ULL,
            0xb9d103fd6854a325ULL,
        },
    },
};

} // namespace goc_test

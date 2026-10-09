// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc_test {

// RX 9070 gfx1201, MODE 0xf0. Both FP16_OVFL settings agree.
// Regenerate using capture_arithmetic_exceptions.py pk_fma/pk_fmac 16.
// 8,192 packed triples per modifier combination; low/high halves differ.
static const uint64_t packed_fma_exception_hashes[2][16] = {
    {
        0xfce7564c3a35d3f1ULL,
        0xfce7564c3a35d3f1ULL,
        0xfce7564c3a35d3f1ULL,
        0xfce7564c3a35d3f1ULL,
        0xb9d103fd6854a325ULL,
        0xb9d103fd6854a325ULL,
        0xb9d103fd6854a325ULL,
        0xb9d103fd6854a325ULL,
        0xfce7564c3a35d3f1ULL,
        0xfce7564c3a35d3f1ULL,
        0xfce7564c3a35d3f1ULL,
        0xfce7564c3a35d3f1ULL,
        0xb9d103fd6854a325ULL,
        0xb9d103fd6854a325ULL,
        0xb9d103fd6854a325ULL,
        0xb9d103fd6854a325ULL,
    },
    {
        0xfce7564c3a35d3f1ULL,
        0xfce7564c3a35d3f1ULL,
        0xfce7564c3a35d3f1ULL,
        0xfce7564c3a35d3f1ULL,
        0xfce7564c3a35d3f1ULL,
        0xfce7564c3a35d3f1ULL,
        0xfce7564c3a35d3f1ULL,
        0xfce7564c3a35d3f1ULL,
        0xfce7564c3a35d3f1ULL,
        0xfce7564c3a35d3f1ULL,
        0xfce7564c3a35d3f1ULL,
        0xfce7564c3a35d3f1ULL,
        0xfce7564c3a35d3f1ULL,
        0xfce7564c3a35d3f1ULL,
        0xfce7564c3a35d3f1ULL,
        0xfce7564c3a35d3f1ULL,
    },
};

} // namespace goc_test

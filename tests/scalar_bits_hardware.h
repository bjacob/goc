// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

// GFX1201 / HIP 7.13; inputs from scalar_integer_inputs, 4096 pairs per op.
// Each digest folds result low/high and SCC. Full, zero, alternating EXEC agree.
// Indexed by incoming SCC then instruction in scalar_bits_names order.
namespace goc_test {

inline const uint64_t scalar_bits_hardware[2][30] = {
    {
        0xe84953f15c993a27ULL, 0x3d4ed6db21c670b6ULL, 0xf6e8336589487825ULL, 0xc962fe4e7fbcb446ULL,
        0x2ea34294c5875fd1ULL, 0x2b8725c816accba9ULL, 0x7ffe735c1aeea8b7ULL, 0xb47a9fd7a2328e0cULL,
        0x678b87ba8579bd7fULL, 0x004e371ab61efe0eULL, 0xb3ec30ac3d29fa4fULL, 0xad6b5843d2753e69ULL,
        0xccbafc96caba454aULL, 0x493c5212bc5bdf04ULL, 0x9059f07ede1786cdULL, 0x42e62b5b5ce24206ULL,
        0x629dea5789a16ba5ULL, 0x14e7f852f035a805ULL, 0x7044f9be2b3226dbULL, 0xacbd3bd497a058e9ULL,
        0xda6b577629d20577ULL, 0x5673e66c1f91e45eULL, 0x714964e4ffd93ca8ULL, 0xf720242ad517c4a9ULL,
        0xa662bffb25105756ULL, 0x8ef438072917e404ULL, 0x7c05d53ef9b347d7ULL, 0xb01072d9abbc01caULL,
        0xf1b4e791ff3cfe2bULL, 0x3780a844b7a249f3ULL,
    },
    {
        0xe84953f15c993a27ULL, 0x3d4ed6db21c670b6ULL, 0xf6e8336589487825ULL, 0xc962fe4e7fbcb446ULL,
        0x2ea34294c5875fd1ULL, 0x2b8725c816accba9ULL, 0x7ffe735c1aeea8b7ULL, 0xb47a9fd7a2328e0cULL,
        0x678b87ba8579bd7fULL, 0x004e371ab61efe0eULL, 0xb3ec30ac3d29fa4fULL, 0xad6b5843d2753e69ULL,
        0xccbafc96caba454aULL, 0x493c5212bc5bdf04ULL, 0x9059f07ede1786cdULL, 0x42e62b5b5ce24206ULL,
        0x629dea5789a16ba5ULL, 0x14e7f852f035a805ULL, 0x84675e3e41373723ULL, 0x36b9b92d4a0ca9fdULL,
        0xda6b577629d20577ULL, 0x5673e66c1f91e45eULL, 0x714964e4ffd93ca8ULL, 0xf720242ad517c4a9ULL,
        0xa662bffb25105756ULL, 0x8ef438072917e404ULL, 0x7c05d53ef9b347d7ULL, 0xb01072d9abbc01caULL,
        0xf1b4e791ff3cfe2bULL, 0x3780a844b7a249f3ULL,
    },
};

} // namespace goc_test

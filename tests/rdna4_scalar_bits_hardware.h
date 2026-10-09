// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

// GFX1201 / HIP 7.13; inputs from scalar_integer_inputs, 4096 pairs per op.
// Each digest folds result low/high and SCC. Full, zero, alternating EXEC agree.
// Indexed by incoming SCC then instruction in scalar_bits_names order.
namespace goc_test {

inline const uint64_t scalar_bits_hardware[2][30] = {
    {
        UINT64_C(0xe84953f15c993a27), UINT64_C(0x3d4ed6db21c670b6), UINT64_C(0xf6e8336589487825),
        UINT64_C(0xc962fe4e7fbcb446), UINT64_C(0x2ea34294c5875fd1), UINT64_C(0x2b8725c816accba9),
        UINT64_C(0x7ffe735c1aeea8b7), UINT64_C(0xb47a9fd7a2328e0c), UINT64_C(0x678b87ba8579bd7f),
        UINT64_C(0x004e371ab61efe0e), UINT64_C(0xb3ec30ac3d29fa4f), UINT64_C(0xad6b5843d2753e69),
        UINT64_C(0xccbafc96caba454a), UINT64_C(0x493c5212bc5bdf04), UINT64_C(0x9059f07ede1786cd),
        UINT64_C(0x42e62b5b5ce24206), UINT64_C(0x629dea5789a16ba5), UINT64_C(0x14e7f852f035a805),
        UINT64_C(0x7044f9be2b3226db), UINT64_C(0xacbd3bd497a058e9), UINT64_C(0xda6b577629d20577),
        UINT64_C(0x5673e66c1f91e45e), UINT64_C(0x714964e4ffd93ca8), UINT64_C(0xf720242ad517c4a9),
        UINT64_C(0xa662bffb25105756), UINT64_C(0x8ef438072917e404), UINT64_C(0x7c05d53ef9b347d7),
        UINT64_C(0xb01072d9abbc01ca), UINT64_C(0xf1b4e791ff3cfe2b), UINT64_C(0x3780a844b7a249f3),
    },
    {
        UINT64_C(0xe84953f15c993a27), UINT64_C(0x3d4ed6db21c670b6), UINT64_C(0xf6e8336589487825),
        UINT64_C(0xc962fe4e7fbcb446), UINT64_C(0x2ea34294c5875fd1), UINT64_C(0x2b8725c816accba9),
        UINT64_C(0x7ffe735c1aeea8b7), UINT64_C(0xb47a9fd7a2328e0c), UINT64_C(0x678b87ba8579bd7f),
        UINT64_C(0x004e371ab61efe0e), UINT64_C(0xb3ec30ac3d29fa4f), UINT64_C(0xad6b5843d2753e69),
        UINT64_C(0xccbafc96caba454a), UINT64_C(0x493c5212bc5bdf04), UINT64_C(0x9059f07ede1786cd),
        UINT64_C(0x42e62b5b5ce24206), UINT64_C(0x629dea5789a16ba5), UINT64_C(0x14e7f852f035a805),
        UINT64_C(0x84675e3e41373723), UINT64_C(0x36b9b92d4a0ca9fd), UINT64_C(0xda6b577629d20577),
        UINT64_C(0x5673e66c1f91e45e), UINT64_C(0x714964e4ffd93ca8), UINT64_C(0xf720242ad517c4a9),
        UINT64_C(0xa662bffb25105756), UINT64_C(0x8ef438072917e404), UINT64_C(0x7c05d53ef9b347d7),
        UINT64_C(0xb01072d9abbc01ca), UINT64_C(0xf1b4e791ff3cfe2b), UINT64_C(0x3780a844b7a249f3),
    },
};

} // namespace goc_test

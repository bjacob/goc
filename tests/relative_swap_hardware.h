// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc_test {

// RX 9070/gfx1201, 2026-10-09, ROCm clang 23; capture_relative_swap.py.
// Word-wise FNV digests of each case in lane/register order.
// Wave32 SHA-256: 3c021eb9a7fb1c5f1972f169c3a265ccaaeda79585fb0cafda4323b12258154c
// Wave64 SHA-256: a2d03d92c4c0cf924903851a6e3ca1ddbba9b1764158f4f9312c4c611caf3162
static const uint64_t relative_swap_hashes[2][24] = {
    {0xbed7e67bd4605be5ULL, 0x785284323af3ad85ULL, 0x50fcc2394c8290a5ULL, 0x6e6491b4b841a3e5ULL,
     0xbed7e67bd4605be5ULL, 0xaf56efab4f82ee75ULL, 0x5b30418a90201e65ULL, 0x9c4a2e548fb444a5ULL,
     0xbed7e67bd4605be5ULL, 0xbed7e67bd4605be5ULL, 0xbed7e67bd4605be5ULL, 0xbed7e67bd4605be5ULL,
     0xbed7e67bd4605be5ULL, 0xbed7e67bd4605be5ULL, 0xbed7e67bd4605be5ULL, 0xbed7e67bd4605be5ULL,
     0xbed7e67bd4605be5ULL, 0xbed7e67bd4605be5ULL, 0xbed7e67bd4605be5ULL, 0xbed7e67bd4605be5ULL,
     0xbed7e67bd4605be5ULL, 0xaf56efab4f82ee75ULL, 0x5b30418a90201e65ULL, 0x9c4a2e548fb444a5ULL},
    {0x85222e9a0e3d6e25ULL, 0x6d41e2a9f6626245ULL, 0xcdc357e6c93b7a25ULL, 0xd3ff4e300042dba5ULL,
     0x85222e9a0e3d6e25ULL, 0x77f2ea098868c435ULL, 0x148b965ed52b92a5ULL, 0x53978d345fee14a5ULL,
     0x85222e9a0e3d6e25ULL, 0x85222e9a0e3d6e25ULL, 0x85222e9a0e3d6e25ULL, 0x85222e9a0e3d6e25ULL,
     0x85222e9a0e3d6e25ULL, 0x85222e9a0e3d6e25ULL, 0x85222e9a0e3d6e25ULL, 0x85222e9a0e3d6e25ULL,
     0x85222e9a0e3d6e25ULL, 0x85222e9a0e3d6e25ULL, 0x85222e9a0e3d6e25ULL, 0x85222e9a0e3d6e25ULL,
     0x85222e9a0e3d6e25ULL, 0x77f2ea098868c435ULL, 0x148b965ed52b92a5ULL, 0x53978d345fee14a5ULL},
};

} // namespace goc_test

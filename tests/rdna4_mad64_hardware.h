// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc_test {

// RX 9070 gfx1201, captured 2026-10-09. 4096 Cartesian triples, A then B then C.
// Digests group by unsigned/signed, none/CLAMP, and full/0x33333333/zero EXEC.
// D starts at 0xdeadbeefcafebabe. FNV-1a consumes each wave's 32 result words
// (8 little-endian bytes each), then its scalar output mask (4 bytes).
static const uint32_t mad64_capture_factors[16] = {
    0x00000000, 0x00000001, 0x00000002, 0x00000003, 0x7fffffff, 0x80000000, 0x80000001, 0xfffffffd,
    0xfffffffe, 0xffffffff, 0x00010000, 0x0000ffff, 0x55555555, 0xaaaaaaaa, 0x12345678, 0xfedcba98};

static const uint64_t mad64_capture_addends[16] = {UINT64_C(0x0),
                                                   UINT64_C(0x1),
                                                   UINT64_C(0x2),
                                                   UINT64_C(0x7fffffffffffffff),
                                                   UINT64_C(0x8000000000000000),
                                                   UINT64_C(0x8000000000000001),
                                                   UINT64_C(0xfffffffffffffffd),
                                                   UINT64_C(0xfffffffffffffffe),
                                                   UINT64_C(0xffffffffffffffff),
                                                   UINT64_C(0xffffffff),
                                                   UINT64_C(0x100000000),
                                                   UINT64_C(0x3fffffffffffffff),
                                                   UINT64_C(0x4000000000000000),
                                                   UINT64_C(0xc000000000000000),
                                                   UINT64_C(0x123456789abcdef),
                                                   UINT64_C(0xfedcba9876543210)};

static const uint64_t mad64_capture_digests[2][2][3] = {
    {
        {UINT64_C(0x62e0069d77fbd9f5), UINT64_C(0x8c92311ad95ffaa5), UINT64_C(0x38a8f6ad5b220b25)},
        {UINT64_C(0x3834f2f66712f5d6), UINT64_C(0xe12215bb422ca8a0), UINT64_C(0x38a8f6ad5b220b25)},
    },
    {
        {UINT64_C(0xd3e48e222cc97653), UINT64_C(0xc8bb4d206fae18be), UINT64_C(0x38a8f6ad5b220b25)},
        {UINT64_C(0x8522468f5505e686), UINT64_C(0xd0a1a128efe8be32), UINT64_C(0x38a8f6ad5b220b25)},
    },
};

} // namespace goc_test

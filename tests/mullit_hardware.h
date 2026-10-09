// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc_test {

// gfx1201, MODE=0xf0, 2026-10-09. 4096 Cartesian A/B/C triples per mode.
// FNV-1a consumes four little-endian result bytes, with NaNs canonicalized to
// 0x7fc00000 because loose semantics leave their payloads unspecified.
static const uint32_t mullit_capture_values[] = {
    0x0,        0x80000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x7f7fffff, 0xff7fffff,
    0x7f800000, 0xff800000, 0x7fc12345, 0xff812345, 0x1,        0x80000001, 0x800000,   0x80800000};

static const uint32_t mullit_capture_modes[] = {
    0x0, 0x1, 0x2, 0x4, 0x38, 0x3f, 0xa, 0x11, 0x40, 0x80, 0xc0, 0x100, 0x140, 0x180, 0x1c0, 0x1ff};

static const uint64_t mullit_capture_digests[] = {
    0xf20466eb7a036845ULL, 0x07c64fed29076f45ULL, 0xb60dd7a82effec45ULL, 0x02aafccdafbd9bc5ULL,
    0x0196140c04f241a5ULL, 0x41eec6ad6eb96325ULL, 0x36ff4f53c9859e45ULL, 0x187d937f707d2fc5ULL,
    0x650a1308f0e11c35ULL, 0xee4837ef37fec675ULL, 0xc52eafd4ceec33e5ULL, 0x3bd560af486e741dULL,
    0xc02d4db4db0465fdULL, 0x436fb9f2af99247dULL, 0xfe5c40e88c10111dULL, 0x9c1bda7f8c872325ULL,
};

} // namespace goc_test

// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

// gfx1201 capture_hwreg.py, MODE and EXCP_FLAG_USER, 20,480 triples.
// SHA256 399b3ca4c4f9eacfe1bc7b94c6d3ff6c575f2ec8855566dae46d972019b0e1a8.
// Hash updated register words for each [register][seed][source] group,
// offset-major then width-minor. Capture preserves/restores the GPU register.
namespace goc_test {

static const uint64_t hwreg_write_hashes[2][2][5] = {
    {
        {0x51d88627df287325ULL, 0x66080f63e628c16bULL, 0xcc705c18b52ee90dULL, 0xa5a37e94c03d8723ULL,
         0x66080f63e628c16bULL},
        {0x7b2ed604ec8e1f6bULL, 0x0579191b3e539725ULL, 0xe5f3642a4181dde3ULL, 0xcd0c00bc9315a66dULL,
         0x0579191b3e539725ULL},
    },
    {
        {0x51d88627df287325ULL, 0xcaefc48692a1916bULL, 0x76cc0c7132fd880dULL, 0xf9f9b58bbdbb2a23ULL,
         0xcaefc48692a1916bULL},
        {0xfc02ba51ef63e86bULL, 0x7860a9489b0b9725ULL, 0x0e2952fa954e60e3ULL, 0x3d0084f7be22456dULL,
         0x7860a9489b0b9725ULL},
    },
};

} // namespace goc_test

// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc_test {

// RX 9070/gfx1201, Wave32 and Wave64, 2026-10-09. Generated with
// capture_ds_swizzle.py and ROCm clang 23. 2,368 offsets x eight EXEC masks
// per wave size. Digests mix each output uint32_t once, in offset/lane order.
// Wave32 capture SHA-256: 98daaa6edb4e5fab0607c16407807f3dcdd8695f1838289300f91900786676be
// Wave64 capture SHA-256: 81b084d2435870b169ade9d0c0a5dce02f79f6c95b70a9e820dcdd06129ec8ee
static const uint64_t ds_swizzle_hashes[2][8] = {
    {0x920e53043fabb535ULL, 0x7b42dfa5a5beab25ULL, 0xed21e6a115ee2545ULL, 0x6bfaea55c089d4bdULL,
     0x50c399c3a69eace4ULL, 0xff23f43888160945ULL, 0xad614b43fdfec3adULL, 0xdc6d31d439c3a8fdULL},
    {0x4728358a698eb585ULL, 0xbfbd0816b5a2d325ULL, 0x4c973b7b543ae3a5ULL, 0xc0a452d74e7e295ULL,
     0x51eabf66573a43a4ULL, 0x534b2e767e3a36e5ULL, 0x3c082664e6a45cb5ULL, 0x6904cdf64e17c935ULL}};

} // namespace goc_test

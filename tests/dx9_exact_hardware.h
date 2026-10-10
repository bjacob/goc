// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc_test {

// RX 9070 gfx1201 RNE. capture_arithmetic_exceptions.py fma_dx9_zero 32,
// with/without --rounding-boundaries. All four coupled denormal modes agree.
// Word-wise FNV over result_lo for the first 16 * 8192 records, no FP16_OVFL.
// [edge-random/boundary]. All 2,097,152 captured exception records were zero.
static const uint64_t dx9_exact_hashes[2] = {0x95314a0e22b7d617ULL, 0x556e5bc1d1f535b5ULL};

} // namespace goc_test

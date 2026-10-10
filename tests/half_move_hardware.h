// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc_test {

// RX 9070/gfx1201, Wave32, 2026-10-09. capture_half_move.py uses raw VOP3
// opcode 412, all ABS/NEG/OMOD/CLAMP and source/destination half selections.
// Word-wise digest of 4,096 outputs; source high half uses input[(lane+7)%32].
// Raw capture SHA-256: c40bc68ccac189c01c0715648abd0d216117329a2d4148331713480e1ef83a7f
static const uint16_t half_move_inputs[32] = {
    0,      0x8000, 1,      0x8001, 0x3ff,  0x83ff, 0x400,  0x8400, 0x3c00, 0xbc00, 0x3800,
    0xb800, 0x4000, 0xc000, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7c01, 0xfc01, 0x7e01, 0xfe01,
    0x3555, 0xb555, 0x3bff, 0xbbff, 0x3c01, 0xbc01, 0x7bfe, 0xfbfe, 0x2aaa, 0xaaaa};
static const uint64_t half_move_hash = 0x662d34d11605cc25ULL;

} // namespace goc_test

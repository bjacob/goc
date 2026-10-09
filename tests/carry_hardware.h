// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc_test {

// RX 9070 gfx1201, captured 2026-10-09. Input carry is 0xa5a5a5a5.
// Instructions: add, sub, subrev, then their CI forms; modifiers: none/CLAMP.
// Each eight-lane input pattern repeats four times in the wave.
static const uint32_t carry_capture_inputs[2][8] = {
    {0, UINT32_MAX, UINT32_MAX, 0, 0, UINT32_MAX, 0x80000000, 0x7fffffff},
    {0, 1, UINT32_MAX, UINT32_MAX, 1, 0, 0x80000000, 1}};

static const uint32_t carry_capture_values[6][2][8] = {
    {
        {0x00000000, 0x00000000, 0xfffffffe, 0xffffffff, 0x00000001, 0xffffffff, 0x00000000,
         0x80000000},
        {0x00000000, 0xffffffff, 0xffffffff, 0xffffffff, 0x00000001, 0xffffffff, 0xffffffff,
         0x80000000},
    },
    {
        {0x00000000, 0xfffffffe, 0x00000000, 0x00000001, 0xffffffff, 0xffffffff, 0x00000000,
         0x7ffffffe},
        {0x00000000, 0xfffffffe, 0x00000000, 0x00000000, 0x00000000, 0xffffffff, 0x00000000,
         0x7ffffffe},
    },
    {
        {0x00000000, 0x00000002, 0x00000000, 0xffffffff, 0x00000001, 0x00000001, 0x00000000,
         0x80000002},
        {0x00000000, 0x00000000, 0x00000000, 0xffffffff, 0x00000001, 0x00000000, 0x00000000,
         0x00000000},
    },
    {
        {0x00000001, 0x00000000, 0xffffffff, 0xffffffff, 0x00000001, 0x00000000, 0x00000000,
         0x80000001},
        {0x00000001, 0xffffffff, 0xffffffff, 0xffffffff, 0x00000001, 0xffffffff, 0xffffffff,
         0x80000001},
    },
    {
        {0xffffffff, 0xfffffffe, 0xffffffff, 0x00000001, 0xffffffff, 0xfffffffe, 0x00000000,
         0x7ffffffd},
        {0x00000000, 0xfffffffe, 0x00000000, 0x00000000, 0x00000000, 0xfffffffe, 0x00000000,
         0x7ffffffd},
    },
    {
        {0xffffffff, 0x00000002, 0xffffffff, 0xffffffff, 0x00000001, 0x00000000, 0x00000000,
         0x80000001},
        {0x00000000, 0x00000000, 0x00000000, 0xffffffff, 0x00000001, 0x00000000, 0x00000000,
         0x00000000},
    },
};

// Scalar output masks for full EXEC, EXEC=0x33333333 and zero EXEC.
static const uint32_t carry_capture_masks[6][3] = {
    {0x46464646, 0x02020202, 0x00000000}, {0x18181818, 0x10101010, 0x00000000},
    {0xa2a2a2a2, 0x22222222, 0x00000000}, {0x66666666, 0x22222222, 0x00000000},
    {0x1d1d1d1d, 0x11111111, 0x00000000}, {0xa7a7a7a7, 0x23232323, 0x00000000},
};

} // namespace goc_test

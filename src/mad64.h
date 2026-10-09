// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// MAD saturation adapted from rocjitsu; signed scalar output follows gfx1201
// captures (the 65-bit sign, rather than rocjitsu's signed-overflow indication).

#pragma once

#include <stdint.h>

namespace goc {

template <bool Signed>
uint32_t mad64_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c);

template <bool Signed>
uint32_t mad64_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c);

} // namespace goc

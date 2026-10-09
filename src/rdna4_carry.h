// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Carry/borrow semantics adapted from rocjitsu's RDNA4 arithmetic handlers.

#pragma once

#include "goc/goc.h"

#include <stdint.h>

namespace goc {

enum class CarryOp { Add, Sub, Subrev };

template <CarryOp Op, bool WithCarry>
uint32_t carry_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b, uint32_t input_carry);

template <CarryOp Op, bool WithCarry>
uint32_t carry_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b, uint32_t input_carry);

} // namespace goc

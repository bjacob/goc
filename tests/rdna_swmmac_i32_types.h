// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Imported from rocjitsu ad2bb973ec4 (fix de4654cfc34).
// Physical gfx1201 captures, 2026-10-10; see upstream float_dot/README.md.

#pragma once

#include <array>
#include <cstdint>

struct RdnaSwmmacIntegerCase {
  uint32_t width;
  uint32_t key;
  uint32_t dst;
  std::array<uint32_t, 256> expected;
};

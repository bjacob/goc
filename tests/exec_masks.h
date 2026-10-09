// SPDX-License-Identifier: MIT

#pragma once

#include <random>
#include <stdint.h>
#include <vector>

// Wave32 masks covering every single active/inactive lane, SIMD chunk boundaries,
// empty/full/alternating masks, deterministic random masks.
inline std::vector<uint32_t> exec_masks() {
  std::vector<uint32_t> masks = {0, UINT32_MAX, 0x55555555U, 0xaaaaaaaaU};
  for (int lane = 0; lane < 32; ++lane) {
    uint32_t bit = 1U << lane;
    masks.push_back(bit);
    masks.push_back(uint32_t(~bit));
  }
  std::mt19937 random(12345);
  for (int i = 0; i < 16; ++i) {
    random(); // Preserve the existing deterministic low-word sequence.
    uint32_t low = random();
    masks.push_back(low);
  }
  return masks;
}

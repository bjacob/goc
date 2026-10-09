// SPDX-License-Identifier: MIT

#pragma once

#include <cstring>
#include <stdint.h>

namespace goc_test {

// Recreate the shared gfx1201 dense/sparse integer WMMA capture inputs.
// Registers 0..5 hold operand words, 6..13 near-limit accumulators, and 14
// sparse metadata (unused by dense WMMA). Seed and draw order define the corpus.
inline void integer_wmma_capture_inputs(unsigned sample, uint32_t (&data)[15][32]) {
  std::memset(data, 0, sizeof(data));
  uint32_t state = 0xa3987412u + sample;
  auto next = [&]() {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
  };
  for (unsigned reg = 0; reg < 6; ++reg)
    for (auto &word : data[reg])
      word = next();
  const uint32_t acc[] = {0, 0x7fffffff, 0x80000000, 0xffffffff, 0x7fffff80, 0x80000080};
  for (unsigned reg = 6; reg < 14; ++reg)
    for (auto &word : data[reg])
      word = acc[next() % 6];
  const unsigned pairs[] = {4, 8, 12, 9, 13, 14};
  for (auto &word : data[14])
    for (unsigned group = 0; group < 8; ++group)
      word |= pairs[next() % 6] << (4 * group);
}

} // namespace goc_test

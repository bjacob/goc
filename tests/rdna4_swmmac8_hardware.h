// SPDX-License-Identifier: MIT

#pragma once

#include <cstring>
#include <stdint.h>

namespace goc_test {

// gfx1201, 2026-10-09, MODE=0xf0. Four FP8/BF8 combinations, sixteen
// deterministic inputs, and both index keys. FNV-1a consumes D register then
// lane, four little-endian bytes per word. Integer-valued inputs keep sums
// exact, without promising bit-exact hardware accumulation for general inputs.
static const uint64_t swmmac8_capture_digests[4][16][2] = {
    {
        {UINT64_C(0x2f442f766c6f0b8a), UINT64_C(0xeb024bd6d2a5e5ab)},
        {UINT64_C(0x4178ef9d7b855b05), UINT64_C(0x05be30a943d8adea)},
        {UINT64_C(0x1f34d66d5a146473), UINT64_C(0xda90caa61b2414e0)},
        {UINT64_C(0xbc51a4a7e6b37c55), UINT64_C(0x638e891556091d1d)},
        {UINT64_C(0x8c25ad4478a31933), UINT64_C(0xbc866c4c0ebc2e52)},
        {UINT64_C(0x136a3518929bcc6a), UINT64_C(0x2268157c5fba6fa3)},
        {UINT64_C(0x55d2b69f022e8648), UINT64_C(0x849a8def729666e2)},
        {UINT64_C(0x9f16c8d5c37b6c3d), UINT64_C(0x83d8283877601a75)},
        {UINT64_C(0xbb5b845c16553ed8), UINT64_C(0x679d1bd23b7c1c12)},
        {UINT64_C(0x98237a4fadf4e003), UINT64_C(0xaac166a328fc6730)},
        {UINT64_C(0x6b63bf1d2f9cc10a), UINT64_C(0x89eea7179919f64a)},
        {UINT64_C(0xd59cb61e414bda7b), UINT64_C(0x4b870df1304f0382)},
        {UINT64_C(0xdea68b9c7a532f42), UINT64_C(0x92e6ee25142e437b)},
        {UINT64_C(0xaae47162f5ce59e3), UINT64_C(0xf56683e9f694bc70)},
        {UINT64_C(0x59cbae7ed0dcf965), UINT64_C(0x9654b8b0e9d3023d)},
        {UINT64_C(0x818913cc1a06d473), UINT64_C(0x3d90195a73d8dbed)},
    },
    {
        {UINT64_C(0x2f442f766c6f0b8a), UINT64_C(0xeb024bd6d2a5e5ab)},
        {UINT64_C(0x4178ef9d7b855b05), UINT64_C(0x05be30a943d8adea)},
        {UINT64_C(0x1f34d66d5a146473), UINT64_C(0xda90caa61b2414e0)},
        {UINT64_C(0xbc51a4a7e6b37c55), UINT64_C(0x638e891556091d1d)},
        {UINT64_C(0x8c25ad4478a31933), UINT64_C(0xbc866c4c0ebc2e52)},
        {UINT64_C(0x136a3518929bcc6a), UINT64_C(0x2268157c5fba6fa3)},
        {UINT64_C(0x55d2b69f022e8648), UINT64_C(0x849a8def729666e2)},
        {UINT64_C(0x9f16c8d5c37b6c3d), UINT64_C(0x83d8283877601a75)},
        {UINT64_C(0xbb5b845c16553ed8), UINT64_C(0x679d1bd23b7c1c12)},
        {UINT64_C(0x98237a4fadf4e003), UINT64_C(0xaac166a328fc6730)},
        {UINT64_C(0x6b63bf1d2f9cc10a), UINT64_C(0x89eea7179919f64a)},
        {UINT64_C(0xd59cb61e414bda7b), UINT64_C(0x4b870df1304f0382)},
        {UINT64_C(0xdea68b9c7a532f42), UINT64_C(0x92e6ee25142e437b)},
        {UINT64_C(0xaae47162f5ce59e3), UINT64_C(0xf56683e9f694bc70)},
        {UINT64_C(0x59cbae7ed0dcf965), UINT64_C(0x9654b8b0e9d3023d)},
        {UINT64_C(0x818913cc1a06d473), UINT64_C(0x3d90195a73d8dbed)},
    },
    {
        {UINT64_C(0x2f442f766c6f0b8a), UINT64_C(0xeb024bd6d2a5e5ab)},
        {UINT64_C(0x4178ef9d7b855b05), UINT64_C(0x05be30a943d8adea)},
        {UINT64_C(0x1f34d66d5a146473), UINT64_C(0xda90caa61b2414e0)},
        {UINT64_C(0xbc51a4a7e6b37c55), UINT64_C(0x638e891556091d1d)},
        {UINT64_C(0x8c25ad4478a31933), UINT64_C(0xbc866c4c0ebc2e52)},
        {UINT64_C(0x136a3518929bcc6a), UINT64_C(0x2268157c5fba6fa3)},
        {UINT64_C(0x55d2b69f022e8648), UINT64_C(0x849a8def729666e2)},
        {UINT64_C(0x9f16c8d5c37b6c3d), UINT64_C(0x83d8283877601a75)},
        {UINT64_C(0xbb5b845c16553ed8), UINT64_C(0x679d1bd23b7c1c12)},
        {UINT64_C(0x98237a4fadf4e003), UINT64_C(0xaac166a328fc6730)},
        {UINT64_C(0x6b63bf1d2f9cc10a), UINT64_C(0x89eea7179919f64a)},
        {UINT64_C(0xd59cb61e414bda7b), UINT64_C(0x4b870df1304f0382)},
        {UINT64_C(0xdea68b9c7a532f42), UINT64_C(0x92e6ee25142e437b)},
        {UINT64_C(0xaae47162f5ce59e3), UINT64_C(0xf56683e9f694bc70)},
        {UINT64_C(0x59cbae7ed0dcf965), UINT64_C(0x9654b8b0e9d3023d)},
        {UINT64_C(0x818913cc1a06d473), UINT64_C(0x3d90195a73d8dbed)},
    },
    {
        {UINT64_C(0x2f442f766c6f0b8a), UINT64_C(0xeb024bd6d2a5e5ab)},
        {UINT64_C(0x4178ef9d7b855b05), UINT64_C(0x05be30a943d8adea)},
        {UINT64_C(0x1f34d66d5a146473), UINT64_C(0xda90caa61b2414e0)},
        {UINT64_C(0xbc51a4a7e6b37c55), UINT64_C(0x638e891556091d1d)},
        {UINT64_C(0x8c25ad4478a31933), UINT64_C(0xbc866c4c0ebc2e52)},
        {UINT64_C(0x136a3518929bcc6a), UINT64_C(0x2268157c5fba6fa3)},
        {UINT64_C(0x55d2b69f022e8648), UINT64_C(0x849a8def729666e2)},
        {UINT64_C(0x9f16c8d5c37b6c3d), UINT64_C(0x83d8283877601a75)},
        {UINT64_C(0xbb5b845c16553ed8), UINT64_C(0x679d1bd23b7c1c12)},
        {UINT64_C(0x98237a4fadf4e003), UINT64_C(0xaac166a328fc6730)},
        {UINT64_C(0x6b63bf1d2f9cc10a), UINT64_C(0x89eea7179919f64a)},
        {UINT64_C(0xd59cb61e414bda7b), UINT64_C(0x4b870df1304f0382)},
        {UINT64_C(0xdea68b9c7a532f42), UINT64_C(0x92e6ee25142e437b)},
        {UINT64_C(0xaae47162f5ce59e3), UINT64_C(0xf56683e9f694bc70)},
        {UINT64_C(0x59cbae7ed0dcf965), UINT64_C(0x9654b8b0e9d3023d)},
        {UINT64_C(0x818913cc1a06d473), UINT64_C(0x3d90195a73d8dbed)},
    },
};

// Reproduce physical inputs from one captured matrix operation.
inline void swmmac8_capture_inputs(unsigned op, unsigned sample, uint32_t (&data)[15][32]) {
  std::memset(data, 0, sizeof(data));
  uint32_t state = 0x7381a593u + sample;
  auto next = [&]() {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
  };
  const uint8_t values[2][5] = {{0xc0, 0xb8, 0, 0x38, 0x40}, {0xc0, 0xbc, 0, 0x3c, 0x40}};
  for (unsigned reg = 0; reg < 6; ++reg)
    for (auto &word : data[reg])
      for (unsigned byte = 0; byte < 4; ++byte)
        word |= uint32_t(values[reg < 2 ? op / 2 : op % 2][next() % 5]) << (8 * byte);
  for (unsigned reg = 6; reg < 14; ++reg)
    for (auto &word : data[reg]) {
      float c = float(int(next() % 5) - 2);
      std::memcpy(&word, &c, sizeof(word));
    }
  const unsigned pairs[] = {4, 8, 12, 9, 13, 14};
  for (auto &word : data[14])
    for (unsigned group = 0; group < 8; ++group)
      word |= pairs[next() % 6] << (4 * group);
}

} // namespace goc_test

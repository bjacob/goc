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
        {0x2f442f766c6f0b8aULL, 0xeb024bd6d2a5e5abULL},
        {0x4178ef9d7b855b05ULL, 0x05be30a943d8adeaULL},
        {0x1f34d66d5a146473ULL, 0xda90caa61b2414e0ULL},
        {0xbc51a4a7e6b37c55ULL, 0x638e891556091d1dULL},
        {0x8c25ad4478a31933ULL, 0xbc866c4c0ebc2e52ULL},
        {0x136a3518929bcc6aULL, 0x2268157c5fba6fa3ULL},
        {0x55d2b69f022e8648ULL, 0x849a8def729666e2ULL},
        {0x9f16c8d5c37b6c3dULL, 0x83d8283877601a75ULL},
        {0xbb5b845c16553ed8ULL, 0x679d1bd23b7c1c12ULL},
        {0x98237a4fadf4e003ULL, 0xaac166a328fc6730ULL},
        {0x6b63bf1d2f9cc10aULL, 0x89eea7179919f64aULL},
        {0xd59cb61e414bda7bULL, 0x4b870df1304f0382ULL},
        {0xdea68b9c7a532f42ULL, 0x92e6ee25142e437bULL},
        {0xaae47162f5ce59e3ULL, 0xf56683e9f694bc70ULL},
        {0x59cbae7ed0dcf965ULL, 0x9654b8b0e9d3023dULL},
        {0x818913cc1a06d473ULL, 0x3d90195a73d8dbedULL},
    },
    {
        {0x2f442f766c6f0b8aULL, 0xeb024bd6d2a5e5abULL},
        {0x4178ef9d7b855b05ULL, 0x05be30a943d8adeaULL},
        {0x1f34d66d5a146473ULL, 0xda90caa61b2414e0ULL},
        {0xbc51a4a7e6b37c55ULL, 0x638e891556091d1dULL},
        {0x8c25ad4478a31933ULL, 0xbc866c4c0ebc2e52ULL},
        {0x136a3518929bcc6aULL, 0x2268157c5fba6fa3ULL},
        {0x55d2b69f022e8648ULL, 0x849a8def729666e2ULL},
        {0x9f16c8d5c37b6c3dULL, 0x83d8283877601a75ULL},
        {0xbb5b845c16553ed8ULL, 0x679d1bd23b7c1c12ULL},
        {0x98237a4fadf4e003ULL, 0xaac166a328fc6730ULL},
        {0x6b63bf1d2f9cc10aULL, 0x89eea7179919f64aULL},
        {0xd59cb61e414bda7bULL, 0x4b870df1304f0382ULL},
        {0xdea68b9c7a532f42ULL, 0x92e6ee25142e437bULL},
        {0xaae47162f5ce59e3ULL, 0xf56683e9f694bc70ULL},
        {0x59cbae7ed0dcf965ULL, 0x9654b8b0e9d3023dULL},
        {0x818913cc1a06d473ULL, 0x3d90195a73d8dbedULL},
    },
    {
        {0x2f442f766c6f0b8aULL, 0xeb024bd6d2a5e5abULL},
        {0x4178ef9d7b855b05ULL, 0x05be30a943d8adeaULL},
        {0x1f34d66d5a146473ULL, 0xda90caa61b2414e0ULL},
        {0xbc51a4a7e6b37c55ULL, 0x638e891556091d1dULL},
        {0x8c25ad4478a31933ULL, 0xbc866c4c0ebc2e52ULL},
        {0x136a3518929bcc6aULL, 0x2268157c5fba6fa3ULL},
        {0x55d2b69f022e8648ULL, 0x849a8def729666e2ULL},
        {0x9f16c8d5c37b6c3dULL, 0x83d8283877601a75ULL},
        {0xbb5b845c16553ed8ULL, 0x679d1bd23b7c1c12ULL},
        {0x98237a4fadf4e003ULL, 0xaac166a328fc6730ULL},
        {0x6b63bf1d2f9cc10aULL, 0x89eea7179919f64aULL},
        {0xd59cb61e414bda7bULL, 0x4b870df1304f0382ULL},
        {0xdea68b9c7a532f42ULL, 0x92e6ee25142e437bULL},
        {0xaae47162f5ce59e3ULL, 0xf56683e9f694bc70ULL},
        {0x59cbae7ed0dcf965ULL, 0x9654b8b0e9d3023dULL},
        {0x818913cc1a06d473ULL, 0x3d90195a73d8dbedULL},
    },
    {
        {0x2f442f766c6f0b8aULL, 0xeb024bd6d2a5e5abULL},
        {0x4178ef9d7b855b05ULL, 0x05be30a943d8adeaULL},
        {0x1f34d66d5a146473ULL, 0xda90caa61b2414e0ULL},
        {0xbc51a4a7e6b37c55ULL, 0x638e891556091d1dULL},
        {0x8c25ad4478a31933ULL, 0xbc866c4c0ebc2e52ULL},
        {0x136a3518929bcc6aULL, 0x2268157c5fba6fa3ULL},
        {0x55d2b69f022e8648ULL, 0x849a8def729666e2ULL},
        {0x9f16c8d5c37b6c3dULL, 0x83d8283877601a75ULL},
        {0xbb5b845c16553ed8ULL, 0x679d1bd23b7c1c12ULL},
        {0x98237a4fadf4e003ULL, 0xaac166a328fc6730ULL},
        {0x6b63bf1d2f9cc10aULL, 0x89eea7179919f64aULL},
        {0xd59cb61e414bda7bULL, 0x4b870df1304f0382ULL},
        {0xdea68b9c7a532f42ULL, 0x92e6ee25142e437bULL},
        {0xaae47162f5ce59e3ULL, 0xf56683e9f694bc70ULL},
        {0x59cbae7ed0dcf965ULL, 0x9654b8b0e9d3023dULL},
        {0x818913cc1a06d473ULL, 0x3d90195a73d8dbedULL},
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

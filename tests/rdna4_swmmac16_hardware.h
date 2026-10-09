// SPDX-License-Identifier: MIT

#pragma once

#include <cstring>
#include <stdint.h>

namespace goc_test {

// gfx1201 captures, 2026-10-09, MODE=0xf0. Each operation has 32 variants:
// compact bits select A-low, B-low, A-high, B-high negation, then index_key.
// FNV-1a consumes physical D registers, then lanes, four little-endian bytes
// per word. Inputs are small integers, so these checks do not assume exact
// hardware agreement for arbitrary floating-point accumulation.
static const uint64_t swmmac16_capture_digests[4][32] = {
    {
        0x74ec4aad31d980d5ULL, 0x6c2fd60914252778ULL, 0x6c2fd60914252778ULL, 0x74ec4aad31d980d5ULL,
        0x62a14dc1b54f8c2dULL, 0xbfc570ef3e04521aULL, 0xbfc570ef3e04521aULL, 0x62a14dc1b54f8c2dULL,
        0x62a14dc1b54f8c2dULL, 0xbfc570ef3e04521aULL, 0xbfc570ef3e04521aULL, 0x62a14dc1b54f8c2dULL,
        0x74ec4aad31d980d5ULL, 0x6c2fd60914252778ULL, 0x6c2fd60914252778ULL, 0x74ec4aad31d980d5ULL,
        0x237dd767ee7e3caaULL, 0x24ba1ff32d6af66dULL, 0x24ba1ff32d6af66dULL, 0x237dd767ee7e3caaULL,
        0x8f8879096ff5bde0ULL, 0xb973d6599b5939b8ULL, 0xb973d6599b5939b8ULL, 0x8f8879096ff5bde0ULL,
        0x8f8879096ff5bde0ULL, 0xb973d6599b5939b8ULL, 0xb973d6599b5939b8ULL, 0x8f8879096ff5bde0ULL,
        0x237dd767ee7e3caaULL, 0x24ba1ff32d6af66dULL, 0x24ba1ff32d6af66dULL, 0x237dd767ee7e3caaULL,
    },
    {
        0x74ec4aad31d980d5ULL, 0x6c2fd60914252778ULL, 0x6c2fd60914252778ULL, 0x74ec4aad31d980d5ULL,
        0x62a14dc1b54f8c2dULL, 0xbfc570ef3e04521aULL, 0xbfc570ef3e04521aULL, 0x62a14dc1b54f8c2dULL,
        0x62a14dc1b54f8c2dULL, 0xbfc570ef3e04521aULL, 0xbfc570ef3e04521aULL, 0x62a14dc1b54f8c2dULL,
        0x74ec4aad31d980d5ULL, 0x6c2fd60914252778ULL, 0x6c2fd60914252778ULL, 0x74ec4aad31d980d5ULL,
        0x237dd767ee7e3caaULL, 0x24ba1ff32d6af66dULL, 0x24ba1ff32d6af66dULL, 0x237dd767ee7e3caaULL,
        0x8f8879096ff5bde0ULL, 0xb973d6599b5939b8ULL, 0xb973d6599b5939b8ULL, 0x8f8879096ff5bde0ULL,
        0x8f8879096ff5bde0ULL, 0xb973d6599b5939b8ULL, 0xb973d6599b5939b8ULL, 0x8f8879096ff5bde0ULL,
        0x237dd767ee7e3caaULL, 0x24ba1ff32d6af66dULL, 0x24ba1ff32d6af66dULL, 0x237dd767ee7e3caaULL,
    },
    {
        0x61abfe5a9cc44745ULL, 0x9ee90db29cc263f8ULL, 0x9ee90db29cc263f8ULL, 0x61abfe5a9cc44745ULL,
        0xf2be843305efaaadULL, 0x09972361bca478e8ULL, 0x09972361bca478e8ULL, 0xf2be843305efaaadULL,
        0xf2be843305efaaadULL, 0x09972361bca478e8ULL, 0x09972361bca478e8ULL, 0xf2be843305efaaadULL,
        0x61abfe5a9cc44745ULL, 0x9ee90db29cc263f8ULL, 0x9ee90db29cc263f8ULL, 0x61abfe5a9cc44745ULL,
        0x1379b30d2ab30d4fULL, 0xbd24d03e33d7ced1ULL, 0xbd24d03e33d7ced1ULL, 0x1379b30d2ab30d4fULL,
        0x24946c88587138caULL, 0xaabdc6c794ad4ba1ULL, 0xaabdc6c794ad4ba1ULL, 0x24946c88587138caULL,
        0x24946c88587138caULL, 0xaabdc6c794ad4ba1ULL, 0xaabdc6c794ad4ba1ULL, 0x24946c88587138caULL,
        0x1379b30d2ab30d4fULL, 0xbd24d03e33d7ced1ULL, 0xbd24d03e33d7ced1ULL, 0x1379b30d2ab30d4fULL,
    },
    {
        0x9644be509dfd8425ULL, 0xd2dce19f9553f64bULL, 0xd2dce19f9553f64bULL, 0x9644be509dfd8425ULL,
        0x39b359fed5db8255ULL, 0xf6e579744ffdcbe2ULL, 0xf6e579744ffdcbe2ULL, 0x39b359fed5db8255ULL,
        0x39b359fed5db8255ULL, 0xf6e579744ffdcbe2ULL, 0xf6e579744ffdcbe2ULL, 0x39b359fed5db8255ULL,
        0x9644be509dfd8425ULL, 0xd2dce19f9553f64bULL, 0xd2dce19f9553f64bULL, 0x9644be509dfd8425ULL,
        0xa242f78f2a2ee32bULL, 0xe7770f7c2d8e8fb8ULL, 0xe7770f7c2d8e8fb8ULL, 0xa242f78f2a2ee32bULL,
        0xa0ac1043b662e62aULL, 0x3de8918281453d6dULL, 0x3de8918281453d6dULL, 0xa0ac1043b662e62aULL,
        0xa0ac1043b662e62aULL, 0x3de8918281453d6dULL, 0x3de8918281453d6dULL, 0xa0ac1043b662e62aULL,
        0xa242f78f2a2ee32bULL, 0xe7770f7c2d8e8fb8ULL, 0xe7770f7c2d8e8fb8ULL, 0xa242f78f2a2ee32bULL,
    },
};

// Reproduce the physical inputs used by the GPU captures.
inline void swmmac16_capture_inputs(unsigned op, uint32_t (&data)[21][32]) {
  std::memset(data, 0, sizeof(data));
  uint32_t state = 0x76895231;
  auto next = [&]() {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
  };
  const uint16_t values[] = {0xc000, uint16_t(op % 2 ? 0xbf80 : 0xbc00), 0,
                             uint16_t(op % 2 ? 0x3f80 : 0x3c00), 0x4000};
  auto pair = [&]() {
    uint32_t lo = values[next() % 5];
    return lo | (uint32_t(values[next() % 5]) << 16);
  };
  for (unsigned r = 0; r < 12; ++r)
    for (auto &word : data[r])
      word = pair();
  for (unsigned r = 0; r < (op >= 2 ? 4u : 8u); ++r)
    for (auto &word : data[12 + r]) {
      if (op >= 2)
        word = pair();
      else {
        float value = float(int(next() % 5) - 2);
        std::memcpy(&word, &value, sizeof(word));
      }
    }
  const uint32_t pairs[] = {4, 8, 12, 9, 13, 14};
  for (auto &word : data[20])
    for (unsigned group = 0; group < 8; ++group)
      word |= pairs[next() % 6] << (4 * group);
}

} // namespace goc_test

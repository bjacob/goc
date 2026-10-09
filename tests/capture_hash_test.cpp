// SPDX-License-Identifier: MIT

#include "capture_hash.h"

#include <gtest/gtest.h>
#include <stdint.h>

TEST(CaptureHash, WordAndByteFormatsRemainDistinct) {
  const uint64_t word = UINT64_C(0x0123456789abcdef);
  EXPECT_EQ(goc_test::capture_hash_word(goc_test::capture_hash_seed, word),
            UINT64_C(0x2c8363b00160c13e));
  EXPECT_EQ(goc_test::capture_hash_bytes(goc_test::capture_hash_seed, word, 4),
            UINT64_C(0x87534de2a496655d));
  EXPECT_EQ(goc_test::capture_hash_bytes(goc_test::capture_hash_seed, word, 8),
            UINT64_C(0x37eb3f3347761c55));
  EXPECT_EQ(goc_test::capture_hash_bytes(goc_test::capture_hash_seed, word, 0),
            goc_test::capture_hash_seed);
}

TEST(CaptureHash, ByteChunksPreserveStreamOrder) {
  uint64_t hash = goc_test::capture_hash_seed;
  hash = goc_test::capture_hash_bytes(hash, UINT32_C(0x89abcdef), 4);
  hash = goc_test::capture_hash_bytes(hash, UINT32_C(0x01234567), 4);
  EXPECT_EQ(hash, UINT64_C(0x37eb3f3347761c55));
}

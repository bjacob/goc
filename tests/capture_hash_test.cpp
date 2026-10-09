// SPDX-License-Identifier: MIT

#include "capture_hash.h"

#include <gtest/gtest.h>
#include <stdint.h>

TEST(CaptureHash, WordAndByteFormatsRemainDistinct) {
  const uint64_t word = 0x0123456789abcdefULL;
  EXPECT_EQ(goc_test::capture_hash_word(goc_test::capture_hash_seed, word), 0x2c8363b00160c13eULL);
  EXPECT_EQ(goc_test::capture_hash_bytes(goc_test::capture_hash_seed, word, 4),
            0x87534de2a496655dULL);
  EXPECT_EQ(goc_test::capture_hash_bytes(goc_test::capture_hash_seed, word, 8),
            0x37eb3f3347761c55ULL);
  EXPECT_EQ(goc_test::capture_hash_bytes(goc_test::capture_hash_seed, word, 0),
            goc_test::capture_hash_seed);
}

TEST(CaptureHash, ByteChunksPreserveStreamOrder) {
  uint64_t hash = goc_test::capture_hash_seed;
  hash = goc_test::capture_hash_bytes(hash, 0x89abcdefU, 4);
  hash = goc_test::capture_hash_bytes(hash, 0x01234567U, 4);
  EXPECT_EQ(hash, 0x37eb3f3347761c55ULL);
}

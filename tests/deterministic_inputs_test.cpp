// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "exec_masks.h"

#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

TEST(DeterministicInputs, ExecMaskStream) {
  // Golden digest calculated independently from MT19937's specified recurrence,
  // seed 12345, and exec_masks' documented fixed-prefix/odd-draw ordering.
  const auto masks = exec_masks();
  ASSERT_EQ(masks.size(), 84U);
  uint64_t hash = goc_test::capture_hash_seed;
  for (uint32_t exec_mask : masks)
    hash = goc_test::capture_hash_word(hash, exec_mask);
  EXPECT_EQ(hash, 0x7d1867b9a02a3982ULL);
}

TEST(DeterministicInputs, DenseMatrixFixtureStream) {
  // dense_golden.h uses 768 draws of minstd_rand(7), mapped to [-2, 2].
  // Independent golden: Python integer recurrence s = 48271*s % 2147483647;
  // each mapped value is encoded modulo 2^32 before the word-wise digest.
  std::minstd_rand random(7);
  uint64_t hash = goc_test::capture_hash_seed;
  for (unsigned i = 0; i < 768; ++i) {
    uint32_t value = uint32_t(int(random() % 5) - 2);
    hash = goc_test::capture_hash_word(hash, value);
  }
  EXPECT_EQ(hash, 0x02665833bab2ab58ULL);
}

// SPDX-License-Identifier: MIT

#include "goc/goc.h"

#include <gtest/gtest.h>
#include <initializer_list>
#include <stdint.h>

TEST(LaneTransfer, EveryIndexAndWrappedIndex) {
  for (int width : {32, 64})
    for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
      for (unsigned index = 0; index < 256; ++index)
        for (uint32_t high : {0U, 0xffffff00U}) {
          uint32_t storage[66];
          for (unsigned i = 0; i < 66; ++i)
            storage[i] = 0x7f800001U + i * 1337;
          uint32_t *v = storage + 1, d = 0, lane = index | high;
          auto read = width == 32 ? goc_v_readlane_b32 : goc_v_readlane_b32_wave64;
          auto write = width == 32 ? goc_v_writelane_b32 : goc_v_writelane_b32_wave64;
          auto flags = semantics | GOC_SEMANTICS_STRICT;
          ASSERT_EQ(read(flags, 0, &d, &v, lane), GOC_SUCCESS);
          EXPECT_EQ(d, storage[1 + (lane % width)]);
          ASSERT_EQ(write(flags, 0, &v, 0xdeadbeef, lane), GOC_SUCCESS);
          for (unsigned i = 0; i < 66; ++i)
            EXPECT_EQ(storage[i], i == 1 + lane % width ? 0xdeadbeefU : 0x7f800001U + i * 1337);
          ASSERT_EQ(read(flags, 0, v, &v, lane), GOC_SUCCESS);
          EXPECT_EQ(v[0], 0xdeadbeefU);
        }
}

TEST(LaneTransfer, FirstActiveLaneAndZeroExec) {
  uint32_t a[64];
  const uint32_t *v = a;
  for (unsigned i = 0; i < 64; ++i)
    a[i] = 0x12340000U + i;
  for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL}) {
    uint32_t d = 0;
    ASSERT_EQ(goc_v_readfirstlane_b32(semantics, 0, 0, &d, &v), GOC_SUCCESS);
    EXPECT_EQ(d, a[0]);
    ASSERT_EQ(goc_v_readfirstlane_b32_wave64(semantics, 0, 0, &d, &v), GOC_SUCCESS);
    EXPECT_EQ(d, a[0]);
    for (unsigned first = 0; first < 64; ++first)
      for (uint64_t exec_mask :
           std::initializer_list<uint64_t>{1ULL << first, UINT64_MAX << first}) {
        ASSERT_EQ(goc_v_readfirstlane_b32_wave64(semantics, exec_mask, 0, &d, &v), GOC_SUCCESS);
        EXPECT_EQ(d, a[first]);
        if (first < 32) {
          ASSERT_EQ(goc_v_readfirstlane_b32(semantics, uint32_t(exec_mask), 0, &d, &v),
                    GOC_SUCCESS);
          EXPECT_EQ(d, a[first]);
        }
      }
  }
  ASSERT_EQ(goc_v_readfirstlane_b32_wave64(0, 1ULL << 63, 0, a, &v), GOC_SUCCESS);
  EXPECT_EQ(a[0], 0x1234003fU);
}

TEST(LaneTransfer, InvalidFlagsBeforeOperandAccess) {
  for (unsigned bit = 0; bit < 64; ++bit) {
    auto mode = 1ULL << bit;
    EXPECT_EQ(goc_v_readfirstlane_b32(0, 0, mode, nullptr, nullptr), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_v_readfirstlane_b32_wave64(0, 0, mode, nullptr, nullptr),
              GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_v_readlane_b32(0, mode, nullptr, nullptr, 0), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_v_readlane_b32_wave64(0, mode, nullptr, nullptr, 0), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_v_writelane_b32(0, mode, nullptr, 0, 0), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_v_writelane_b32_wave64(0, mode, nullptr, 0, 0), GOC_ERROR_INVALID_FLAGS);
  }
}

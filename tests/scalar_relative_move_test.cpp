// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "scalar_relative_hardware.h"

#include <algorithm>
#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>

TEST(ScalarRelativeMove, HardwareOffsetsAndRegisterWidths) {
  for (uint64_t flags : {uint64_t(0), GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT})
    for (unsigned op = 0; op < 5; ++op)
      for (unsigned n = 0; n < 5; ++n) {
        uint32_t sgpr[124] = {};
        sgpr[0] = 0x11223344;
        sgpr[1] = 0x55667788;
        for (unsigned r = 64; r < 82; ++r)
          sgpr[r] = 0x12340000 + r - 64;
        uint32_t m0 = n < 4 ? n * 2 : 256;
        uint64_t pair = 0;
        switch (op) {
        case 0:
          ASSERT_EQ(goc_s_movrels_b32(flags, 0, sgpr + 80, sgpr, 124, 64, m0), GOC_SUCCESS);
          break;
        case 1:
          ASSERT_EQ(goc_s_movrels_b64(flags, 0, &pair, sgpr, 124, 64, m0), GOC_SUCCESS);
          sgpr[80] = uint32_t(pair);
          sgpr[81] = uint32_t(pair >> 32);
          break;
        case 2:
          ASSERT_EQ(goc_s_movreld_b32(flags, 0, sgpr, sgpr[80], 124, 64, m0), GOC_SUCCESS);
          break;
        case 3:
          pair = sgpr[80] | uint64_t(sgpr[81]) << 32;
          ASSERT_EQ(goc_s_movreld_b64(flags, 0, sgpr, pair, 124, 64, m0), GOC_SUCCESS);
          break;
        case 4:
          m0 |= (n < 4 ? 6 - 2 * n : 6) << 16;
          ASSERT_EQ(goc_s_movrelsd_2_b32(flags, 0, sgpr, sgpr, 124, 124, 64, 64, m0), GOC_SUCCESS);
          break;
        }
        for (unsigned r = 0; r < 18; ++r)
          EXPECT_EQ(sgpr[64 + r], scalar_relative_hardware[5 * op + n][r])
              << op << '/' << n << '/' << r;
      }
}

TEST(ScalarRelativeMove, SourceBoundsRegionsAndFullM0) {
  uint32_t a[128];
  for (unsigned r = 0; r < 128; ++r)
    a[r] = 0x13570000 + r;
  for (uint32_t count : {0U, 1U, 2U, 108U, 124U, 128U})
    for (uint32_t base : {0U, 2U, 104U, 106U, 108U, 120U, 122U, 124U, UINT32_MAX - 1})
      for (uint32_t offset : {0U, 2U, 4U, 16U, 256U, 65536U, UINT32_MAX - 1}) {
        uint64_t index = uint64_t(base) + offset;
        unsigned region_end = base < 108 ? 108 : base < 124 ? 124 : 0;
        unsigned index32 = index < region_end && index < count ? unsigned(index) : 0;
        unsigned index64 = index + 1 < region_end && index + 1 < count ? unsigned(index) : 0;
        uint32_t d32 = 0;
        uint64_t d64 = 0;
        ASSERT_EQ(goc_s_movrels_b32(GOC_SEMANTICS_EXACT_EMPIRICAL, 0, &d32, count ? a : nullptr,
                                    count, base, offset),
                  GOC_SUCCESS);
        ASSERT_EQ(goc_s_movrels_b64(GOC_SEMANTICS_EXACT_EMPIRICAL, 0, &d64, count ? a : nullptr,
                                    count, base, offset),
                  GOC_SUCCESS);
        EXPECT_EQ(d32, count ? a[index32] : 0U);
        uint64_t expected = count ? a[index64] : 0;
        if (count > 1)
          expected |= uint64_t(a[index64 + 1]) << 32;
        EXPECT_EQ(d64, expected);
      }
}

TEST(ScalarRelativeMove, DestinationBoundsAndSplitFields) {
  for (uint32_t count : {0U, 108U, 124U, 128U})
    for (uint32_t base : {0U, 106U, 108U, 122U, 124U, UINT32_MAX - 1})
      for (uint32_t offset : {0U, 2U, 4U, 256U, UINT32_MAX - 1})
        for (unsigned words : {1U, 2U}) {
          uint32_t data[128], expected[128];
          for (unsigned r = 0; r < 128; ++r)
            data[r] = expected[r] = r;
          uint64_t index = uint64_t(base) + offset;
          unsigned region_end = base < 108 ? 108 : base < 124 ? 124 : 0;
          if (index + words <= count && index + words <= region_end) {
            expected[index] = 0x89abcdef;
            if (words == 2)
              expected[index + 1] = 0x12345678;
          }
          int result = words == 1 ? goc_s_movreld_b32(0, 0, count ? data : nullptr, 0x89abcdef,
                                                      count, base, offset)
                                  : goc_s_movreld_b64(0, 0, count ? data : nullptr,
                                                      0x1234567889abcdefULL, count, base, offset);
          ASSERT_EQ(result, GOC_SUCCESS);
          EXPECT_EQ(std::memcmp(data, expected, sizeof data), 0);
        }
  // Reserved M0 bits are ignored in the split-offset form. Distinct source and
  // destination windows overlap, so the original source value must be read first.
  uint32_t data[124];
  for (unsigned r = 0; r < 124; ++r)
    data[r] = r;
  ASSERT_EQ(goc_s_movrelsd_2_b32(0, 0, data, data, 124, 124, 8, 2, 0xfc00fc00U | (4U << 16) | 6U),
            GOC_SUCCESS);
  EXPECT_EQ(data[12], 8U);
  ASSERT_EQ(goc_s_movrelsd_2_b32(0, 0, data, data, 124, 124, 4, 106, 2), GOC_SUCCESS);
  EXPECT_EQ(data[4], data[0]);
  ASSERT_EQ(goc_s_movrelsd_2_b32(0, 0, nullptr, nullptr, 0, 0, 0, 0, 0), GOC_SUCCESS);
}

TEST(ScalarRelativeMove, ValidationBeforeOperandAccess) {
  uint32_t d32 = 123;
  uint64_t d64 = 456;
  for (unsigned bit = 0; bit < 64; ++bit) {
    uint64_t mode = 1ULL << bit;
    EXPECT_EQ(goc_s_movrels_b32(0, mode, &d32, nullptr, 124, 0, 0), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_s_movrels_b64(0, mode, &d64, nullptr, 124, 0, 0), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_s_movreld_b32(0, mode, nullptr, 0, 124, 0, 0), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_s_movreld_b64(0, mode, nullptr, 0, 124, 0, 0), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_s_movrelsd_2_b32(0, mode, nullptr, nullptr, 124, 124, 0, 0, 0),
              GOC_ERROR_INVALID_FLAGS);
  }
  EXPECT_EQ(d32, 123U);
  EXPECT_EQ(d64, 456U);
}

// SPDX-License-Identifier: MIT

#include "dpp_reference.h"
#include "goc/goc.h"

#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>

TEST(RelativeMove, IndicesBoundsDppAndAliases) {
  const decltype(&goc_v_movrels_b32) functions[] = {goc_v_movrels_b32, goc_v_movreld_b32,
                                                    goc_v_movrelsd_b32, goc_v_movrelsd_2_b32};
  for (unsigned op = 0; op < 4; ++op)
    for (uint32_t m0 : {0U, 1U, 5U, 6U, 1023U, 1024U, 0xffffffffU, 0x00020001U, 0xfc000400U})
      for (uint32_t exec_mask : {0U, UINT32_MAX, 0xaaaaaaaaU, 0x80010001U})
        for (int routing = -1; routing < 7; ++routing)
          for (unsigned base : {0U, 2U}) {
            uint32_t data[8][34], wanted[8][34];
            uint32_t *v[8];
            for (unsigned r = 0; r < 8; ++r) {
              v[r] = data[7 - r] + 1;
              for (unsigned lane = 0; lane < 34; ++lane)
                data[r][lane] = 0xabcdef00U + r * 100 + lane;
            }
            std::memcpy(wanted, data, sizeof data);
            uint64_t src = op == 1 ? 0 : op == 3 ? m0 & 1023 : m0;
            uint64_t dst = op == 0 ? 0 : op == 3 ? (m0 >> 16) & 1023 : m0;
            uint64_t mode = routing < 0 ? 0 : goc_test::dpp_modes[routing];
            src += 1;
            dst += base;
            if (dst < 8)
              for (unsigned lane = 0; lane < 32; ++lane) {
                int source = lane;
                bool write = routing < 0 ? ((exec_mask >> lane) & 1) != 0
                                         : goc_test::dpp_source(mode, exec_mask, lane, source);
                if (write)
                  wanted[7 - dst][lane + 1] = source >= 0 ? v[src < 8 ? src : 0][source] : 0;
              }
            ASSERT_EQ(functions[op](GOC_SEMANTICS_EXACT_EMPIRICAL, exec_mask, mode, v, v, 8, 8,
                                    base, 1, m0),
                      GOC_SUCCESS);
            EXPECT_EQ(std::memcmp(wanted, data, sizeof data), 0)
                << op << "/" << m0 << "/" << routing;
          }
}

TEST(RelativeMove, EmptyRangesAndValidation) {
  const decltype(&goc_v_movrels_b32) functions[] = {goc_v_movrels_b32, goc_v_movreld_b32,
                                                    goc_v_movrelsd_b32, goc_v_movrelsd_2_b32};
  for (auto fn : functions) {
    EXPECT_EQ(fn(0, UINT32_MAX, 0, nullptr, nullptr, 0, 0, 0, 0, 0), GOC_SUCCESS);
    uint32_t data[32];
    uint32_t *v = data;
    for (auto &x : data)
      x = 123;
    ASSERT_EQ(fn(0, UINT32_MAX, 0, &v, nullptr, 1, 0, 0, 0, 0), GOC_SUCCESS);
    for (auto x : data)
      EXPECT_EQ(x, 0U);
    for (unsigned bit = 0; bit < 32; ++bit)
      EXPECT_EQ(fn(0, 0, 1ULL << bit, nullptr, nullptr, 0, 0, 0, 0, 0), GOC_ERROR_INVALID_FLAGS);
  }
}

TEST(RelativeMove, ManualOffsetLimitAndWidenedAddressing) {
  uint32_t data[300][32];
  uint32_t *v[300];
  for (unsigned r = 0; r < 300; ++r) {
    v[r] = data[r];
    for (auto &x : data[r])
      x = r + 100;
  }
  ASSERT_EQ(goc_v_movrels_b32(0, 1, 0, v, v, 300, 300, 1, 0, 255), GOC_SUCCESS);
  EXPECT_EQ(data[1][0], 355U);
  ASSERT_EQ(goc_v_movrels_b32(0, 1, 0, v, v, 300, 300, 1, 0, 256), GOC_SUCCESS);
  EXPECT_EQ(data[1][0], 100U);
  ASSERT_EQ(goc_v_movreld_b32(0, 1, 0, v, v, 300, 300, 0, 0, 256), GOC_SUCCESS);
  EXPECT_EQ(data[256][0], 356U);
  ASSERT_EQ(goc_v_movrels_b32(0, 1, 0, v, v, 300, 300, 1, UINT32_MAX, 3), GOC_SUCCESS);
  EXPECT_EQ(data[1][0], 100U);
  ASSERT_EQ(goc_v_movreld_b32(0, 1, 0, v, v, 300, 300, UINT32_MAX, 2, 1), GOC_SUCCESS);
  EXPECT_EQ(data[0][0], 100U);
}

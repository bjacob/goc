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
      if (!((1ULL << bit) & (GOC_ALU_ABS_A | GOC_ALU_NEG_A))) {
        EXPECT_EQ(fn(0, 0, 1ULL << bit, nullptr, nullptr, 0, 0, 0, 0, 0), GOC_ERROR_INVALID_FLAGS);
      }
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

TEST(RelativeMove, Wave64IndicesDppAndUpperLaneMasks) {
  const decltype(&goc_v_movrels_b32_wave64) functions[] = {
      goc_v_movrels_b32_wave64, goc_v_movreld_b32_wave64, goc_v_movrelsd_b32_wave64,
      goc_v_movrelsd_2_b32_wave64};
  for (unsigned op = 0; op < 4; ++op)
    for (uint64_t exec_mask : {0ULL, 1ULL << 63, 0xaaaaaaaaaaaaaaaaULL, ~0ULL})
      for (uint32_t m0 : {0U, 1U, 6U, 256U, 1023U, 0x40002U, 0xfc04fc02U, UINT32_MAX})
        for (int routing = -1; routing < 7; ++routing) {
          uint32_t data[8][64], expected[8][64];
          uint32_t *v[8];
          for (unsigned r = 0; r < 8; ++r) {
            v[r] = data[r];
            for (unsigned lane = 0; lane < 64; ++lane)
              data[r][lane] = expected[r][lane] = r * 1000 + lane;
          }
          uint64_t src_offset = op == 1 ? 0 : op == 3 ? m0 & 1023 : m0;
          uint64_t dst_offset = op == 0 ? 0 : op == 3 ? (m0 >> 16) & 1023 : m0;
          uint64_t src = 1 + src_offset, dst = 2 + dst_offset;
          if (src >= 8 || src_offset > 255)
            src = 0;
          uint64_t mode = routing < 0 ? 0 : goc_test::dpp_modes[routing];
          if (dst < 8 && dst_offset <= 255)
            for (unsigned lane = 0; lane < 64; ++lane) {
              int source = int(lane);
              bool write = routing < 0 ? (exec_mask >> lane) & 1
                                       : goc_test::dpp_source(mode, exec_mask, lane, source);
              if (write)
                expected[dst][lane] = source < 0 ? 0 : data[src][source];
            }
          ASSERT_EQ(
              functions[op](GOC_SEMANTICS_EXACT_EMPIRICAL, exec_mask, mode, v, v, 8, 8, 2, 1, m0),
              GOC_SUCCESS);
          EXPECT_EQ(std::memcmp(data, expected, sizeof data), 0)
              << op << '/' << m0 << '/' << routing;
        }
  for (auto fn : functions) {
    EXPECT_EQ(fn(0, UINT64_MAX, 0, nullptr, nullptr, 0, 0, 0, 0, 0), GOC_SUCCESS);
    EXPECT_EQ(fn(0, UINT64_MAX, 1ULL << 63, nullptr, nullptr, 256, 256, 0, 0, 0),
              GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(fn(0, 0, 0, nullptr, nullptr, 256, 256, 0, 0, 0), GOC_SUCCESS);
  }
}

TEST(RelativeMove, SignModifiersAfterRegisterSelectionAndDpp) {
  const decltype(&goc_v_movrels_b32) wave32[] = {goc_v_movrels_b32, goc_v_movreld_b32,
                                                 goc_v_movrelsd_b32, goc_v_movrelsd_2_b32};
  const decltype(&goc_v_movrels_b32_wave64) wave64[] = {
      goc_v_movrels_b32_wave64, goc_v_movreld_b32_wave64, goc_v_movrelsd_b32_wave64,
      goc_v_movrelsd_2_b32_wave64};
  const uint32_t patterns[] = {0,           0x80000000U, 1,           0x80000001U,
                               0x7f800000U, 0xff800000U, 0x7f800001U, 0xff800001U,
                               0x7fc12345U, 0xffc12345U, 0x3f800000U, 0xbf800000U};
  for (unsigned lanes : {32U, 64U})
    for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
      for (unsigned op = 0; op < 5; ++op)
        for (unsigned modifiers = 0; modifiers < 4; ++modifiers)
          for (int routing = -1; routing < 7; ++routing)
            for (uint64_t exec_mask : {0ULL, 1ULL << 63, 0xaaaaaaaaaaaaaaaaULL, ~0ULL}) {
              uint32_t storage[3][66], before[3][66];
              uint32_t *v[3];
              for (unsigned r = 0; r < 3; ++r) {
                v[r] = storage[r] + 1;
                for (unsigned lane = 0; lane < 66; ++lane)
                  before[r][lane] = storage[r][lane] = patterns[(lane + r) % 12];
              }
              uint64_t mode = routing < 0 ? 0 : goc_test::dpp_modes[routing];
              mode |= (modifiers & 1 ? GOC_ALU_ABS_A : 0) | (modifiers & 2 ? GOC_ALU_NEG_A : 0);
              uint32_t m0 = op == 3 ? 0x00010001U : 1;
              unsigned dst = op == 0 ? 0 : 1, src = op == 1 || op == 4 ? 1 : 2;
              uint64_t flags = semantics | GOC_SEMANTICS_STRICT;
              int status;
              if (op == 4) // Ordinary MOV, with an in-place source.
                status = lanes == 32 ? goc_v_mov_b32(flags, uint32_t(exec_mask), mode, v + 1, v + 1)
                                     : goc_v_mov_b32_wave64(flags, exec_mask, mode, v + 1, v + 1);
              else
                status = lanes == 32
                             ? wave32[op](flags, uint32_t(exec_mask), mode, v, v, 3, 3, 0, 1, m0)
                             : wave64[op](flags, exec_mask, mode, v, v, 3, 3, 0, 1, m0);
              ASSERT_EQ(status, GOC_SUCCESS);
              for (unsigned r = 0; r < 3; ++r)
                for (unsigned word = 0; word < 66; ++word) {
                  uint32_t expected = before[r][word];
                  if (r == dst && word > 0 && word <= lanes) {
                    unsigned lane = word - 1;
                    int source = lane;
                    bool write = routing < 0 ? bool((exec_mask >> lane) & 1)
                                             : goc_test::dpp_source(mode, exec_mask, lane, source);
                    if (write) {
                      expected = source < 0 ? 0 : before[src][source + 1];
                      if (modifiers & 1)
                        expected &= 0x7fffffffU;
                      if (modifiers & 2)
                        expected ^= 0x80000000U;
                    }
                  }
                  ASSERT_EQ(storage[r][word], expected)
                      << lanes << '/' << op << '/' << modifiers << '/' << routing << '/' << word;
                }
            }
}

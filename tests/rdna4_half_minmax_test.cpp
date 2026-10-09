// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "goc/goc.h"
#include "rdna4_exec_masks.h"
#include "rdna4_half_minmax_reference.h"
#include "rdna4_half_reference.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <random>
#include <stdint.h>

namespace {

const uint16_t values[] = {0,      0x8000, 1,      0x8001, 0x3ff,  0x400,  0x3800, 0xb800,
                           0x3bff, 0x3c00, 0x3e00, 0xbe00, 0x4100, 0xc100, 0x4200, 0x4400,
                           0x4c00, 0xcc00, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7c01, 0xfe00};

void check(uint32_t actual, uint32_t before, uint16_t want, uint32_t mode) {
  int shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  EXPECT_EQ((actual ^ before) & ~(0xffffU << shift), 0u);
  uint16_t got = uint16_t(actual >> shift);
  if (goc_test::half_minmax_nan(want)) {
    EXPECT_TRUE(goc_test::half_minmax_nan(got));
  } else {
    EXPECT_EQ(got, want);
  }
}

} // namespace

TEST(HalfMinmax3, BoundaryCartesianProductsAndEveryEncoding) {
  for (int op = 0; op < 9; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool high : {false, true}) {
        std::mt19937 random(472);
        uint32_t mode =
            high ? GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_C | GOC_ALU_HIGH_D : 0;
        for (unsigned base = 0; base < 13824 + 65536; base += 32) {
          SCOPED_TRACE(::testing::Message() << op << '/' << cpu << '/' << high << '/' << base);
          uint32_t words[4][32];
          uint32_t *p[] = {words[0], words[1], words[2], words[3]};
          for (int lane = 0; lane < 32; ++lane) {
            unsigned i = base + lane;
            for (int reg = 0; reg < 3; ++reg) {
              uint16_t code = base < 13824 ? values[i % 24]
                              : reg == 0   ? uint16_t(i - 13824)
                                           : uint16_t(random());
              if (base < 13824)
                i /= 24;
              words[reg][lane] = high ? (uint32_t(code) << 16) | 0xbeef : 0xdead0000 | code;
            }
            words[3][lane] = 0xdeadbeef;
          }
          ASSERT_EQ(
              goc_test::half_minmax_functions[op](cpu, UINT32_MAX, mode, p + 3, p, p + 1, p + 2),
              GOC_SUCCESS);
          for (int lane = 0; lane < 32; ++lane)
            check(words[3][lane], 0xdeadbeef,
                  goc_test::half_minmax_reference(op, words[0][lane], words[1][lane],
                                                  words[2][lane], mode, false),
                  mode);
        }
      }
}

TEST(HalfMinmax3, AllModifiersAndHalfSelectors) {
  for (int op = 0; op < 9; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint32_t mode = 0; mode < 8192; ++mode)
        for (bool saturate : {false, true}) {
          SCOPED_TRACE(::testing::Message() << op << '/' << cpu << '/' << mode << '/' << saturate);
          uint32_t words[4][32], before[4][32];
          for (int reg = 0; reg < 4; ++reg)
            for (int lane = 0; lane < 32; ++lane)
              words[reg][lane] = values[(lane + reg * 5) % 24] |
                                 (uint32_t(values[(lane * 7 + reg * 3) % 24]) << 16);
          std::memcpy(before, words, sizeof(words));
          int dest = mode % 4;
          uint32_t *p[] = {words[0], words[1], words[2], words[3]};
          ASSERT_EQ(goc_test::half_minmax_functions[op](cpu | (saturate ? GOC_FP16_OVFL : 0),
                                                        UINT32_MAX, mode, p + dest, p, p + 1,
                                                        p + 2),
                    GOC_SUCCESS);
          for (int reg = 0; reg < 4; ++reg)
            for (int lane = 0; lane < 32; ++lane) {
              if (reg == dest) {
                check(words[reg][lane], before[reg][lane],
                      goc_test::half_minmax_reference(op, before[0][lane], before[1][lane],
                                                      before[2][lane], mode, saturate),
                      mode);
              } else {
                EXPECT_EQ(words[reg][lane], before[reg][lane]);
              }
            }
        }
}

TEST(HalfMinmax3, MasksAndAllWholeRegisterAliases) {
  const int layouts[][3] = {{0, 1, 2}, {0, 0, 2}, {0, 1, 0}, {0, 1, 1}, {0, 0, 0}};
  const uint32_t modes[] = {
      0, 63U | GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_C | GOC_ALU_HIGH_D,
      GOC_ALU_HIGH_A | GOC_ALU_HIGH_C | GOC_ALU_HIGH_D | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP,
      GOC_ALU_HIGH_B | GOC_ALU_OMOD_4};
  for (int op = 0; op < 9; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (auto mode : modes)
        for (auto mask : rdna4_exec_masks())
          for (const auto &layout : layouts)
            for (int dest = 0; dest < 4; ++dest) {
              SCOPED_TRACE(::testing::Message()
                           << op << '/' << cpu << '/' << mode << '/' << mask << '/' << dest << '/'
                           << layout[0] << layout[1] << layout[2]);
              uint32_t words[4][34], before[4][34];
              for (int reg = 0; reg < 4; ++reg) {
                std::fill(words[reg], words[reg] + 34, 0xdeadbeef);
                for (int lane = 1; lane <= 32; ++lane)
                  words[reg][lane] = values[(lane + reg * 5) % 24] |
                                     (uint32_t(values[(lane * 7 + reg * 3) % 24]) << 16);
              }
              std::memcpy(before, words, sizeof(words));
              uint32_t *p[] = {words[0] + 1, words[1] + 1, words[2] + 1, words[3] + 1};
              ASSERT_EQ(goc_test::half_minmax_functions[op](
                            cpu, mask, mode, p + dest, p + layout[0], p + layout[1], p + layout[2]),
                        GOC_SUCCESS);
              for (int reg = 0; reg < 4; ++reg)
                for (int lane = 0; lane < 34; ++lane) {
                  if (reg == dest && lane >= 1 && lane <= 32 && ((mask >> (lane - 1)) & 1)) {
                    check(words[reg][lane], before[reg][lane],
                          goc_test::half_minmax_reference(op, before[layout[0]][lane],
                                                          before[layout[1]][lane],
                                                          before[layout[2]][lane], mode, false),
                          mode);
                  } else {
                    EXPECT_EQ(words[reg][lane], before[reg][lane]);
                  }
                }
            }
}

TEST(HalfMinmax3, LiteralOrderNanAndSignedZeroRules) {
  struct Case {
    uint16_t a, b, c, expected[9];
  };

  const Case cases[] = {
      {0x3c00,
       0x4200,
       0x4000,
       {0x3c00, 0x4200, 0x4000, 0x4000, 0x3c00, 0x4200, 0x4000, 0x4000, 0x4000}},
      {0x4200,
       0x3c00,
       0x4000,
       {0x3c00, 0x4200, 0x4000, 0x4000, 0x3c00, 0x4200, 0x4000, 0x4000, 0x4000}},
      {0x7c01,
       0x3c00,
       0x4000,
       {0x3c00, 0x4000, 0x4000, 0x3c00, 0x7e00, 0x7e00, 0x7e00, 0x7e00, 0x3c00}},
      {0x3c00,
       0xfe00,
       0x4000,
       {0x3c00, 0x4000, 0x4000, 0x3c00, 0x7e00, 0x7e00, 0x7e00, 0x7e00, 0x3c00}},
      {0x3c00,
       0x4000,
       0xfe01,
       {0x3c00, 0x4000, 0x3c00, 0x4000, 0x7e00, 0x7e00, 0x7e00, 0x7e00, 0x3c00}},
      {0x8000, 0, 0x8000, {0x8000, 0, 0x8000, 0x8000, 0x8000, 0, 0x8000, 0x8000, 0x8000}},
      {0, 0x8000, 0x8000, {0x8000, 0, 0x8000, 0x8000, 0x8000, 0, 0x8000, 0x8000, 0x8000}}};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (const auto &test : cases)
      for (int op = 0; op < 9; ++op) {
        uint32_t words[4][32];
        uint32_t *p[] = {words[0], words[1], words[2], words[3]};
        std::fill(words[0], words[0] + 32, test.a);
        std::fill(words[1], words[1] + 32, test.b);
        std::fill(words[2], words[2] + 32, test.c);
        std::fill(words[3], words[3] + 32, 0xdeadbeef);
        ASSERT_EQ(goc_test::half_minmax_functions[op](cpu, UINT32_MAX, 0, p + 3, p, p + 1, p + 2),
                  GOC_SUCCESS);
        for (auto word : words[3])
          check(word, 0xdeadbeef, test.expected[op], 0);
      }
}

TEST(HalfMinmax3, ValidationAndSemantics) {
  for (auto fn : goc_test::half_minmax_functions) {
    uint32_t data[32];
    std::fill(data, data + 32, 0xdeadbeef);
    auto p = data;
    for (uint32_t mask : {0U, UINT32_MAX}) {
      for (int bit = 13; bit < 32; ++bit)
        EXPECT_EQ(fn(0, mask, 1U << bit, &p, &p, &p, &p), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(1ULL << 63, mask, 0, &p, &p, &p, &p), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, mask, 0, &p, &p, &p, &p),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
    }
    for (auto word : data)
      EXPECT_EQ(word, 0xdeadbeef);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, 0, &p, &p, &p, &p), GOC_SUCCESS);
  }
}

TEST(HalfMinmax3, HardwareMedianSignedZeroOrdering) {
  const uint32_t values[] = {0, 0x8000, 0x3c00, 0xbc00, 1, 0x8001, 0x7c00, 0xfc00};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint64_t hash = goc_test::capture_hash_seed;
    for (unsigned base = 0; base < 512; base += 32) {
      uint32_t words[4][32];
      for (unsigned lane = 0; lane < 32; ++lane) {
        unsigned index = base + lane;
        words[0][lane] = values[index % 8];
        words[1][lane] = values[(index / 8) % 8];
        words[2][lane] = values[(index / 64) % 8];
        words[3][lane] = 0xdead0000u + lane;
      }
      uint32_t *p[] = {words[0], words[1], words[2], words[3]};
      ASSERT_EQ(goc_rdna4_v_med3_num_f16(cpu, UINT32_MAX, 0, p + 3, p, p + 1, p + 2), GOC_SUCCESS);
      for (uint32_t word : words[3])
        hash = goc_test::capture_hash_word(hash, word);
    }
    EXPECT_EQ(hash, 0x6621dd1b4ac695edULL);
  }
}

TEST(HalfMinmax3, HardwareMedianNanRules) {
  const uint32_t values[] = {0, 0x8000, 0x7c01, 0x7e00, 0xfc01, 0xfe00, 0x3c00, 0xbc00};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint64_t hash = goc_test::capture_hash_seed;
    for (unsigned base = 0; base < 512; base += 32) {
      uint32_t words[4][32];
      for (unsigned lane = 0; lane < 32; ++lane) {
        unsigned index = base + lane;
        words[0][lane] = values[index % 8];
        words[1][lane] = values[(index / 8) % 8];
        words[2][lane] = values[(index / 64) % 8];
        words[3][lane] = 0xdead0000u + lane;
      }
      uint32_t *p[] = {words[0], words[1], words[2], words[3]};
      ASSERT_EQ(goc_rdna4_v_med3_num_f16(cpu, UINT32_MAX, 0, p + 3, p, p + 1, p + 2), GOC_SUCCESS);
      for (uint32_t word : words[3]) {
        if ((word & 0x7fff) > 0x7c00)
          word = (word & 0xffff0000u) | 0x7e00;
        hash = goc_test::capture_hash_word(hash, word);
      }
    }
    EXPECT_EQ(hash, 0x513891779d603325ULL);
  }
}

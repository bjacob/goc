// SPDX-License-Identifier: MIT

#include "exec_masks.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "integer_mad_reference.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

TEST(IntegerMad, BoundaryTriplesAndRandomInputsWithEveryModifier) {
  const uint32_t edge[] = {0,          1,          2,          0x7fff,    0x8000,    0xffff,
                           0x10000,    0x7fffff,   0x800000,   0xffffff,  0x1000000, 0x7ffffffe,
                           0x7fffffff, 0x80000000, 0xfffffffe, 0xffffffff};
  for (int op = 0; op < 4; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int mode = 0; mode < (op < 2 ? 8 : 2); ++mode) {
        SCOPED_TRACE(::testing::Message() << op << "/" << cpu << "/" << mode);
        std::mt19937 random(281);
        for (int start = 0; start < 8192; start += 32) {
          uint32_t words[4][32];
          uint32_t *p[] = {words[0], words[1], words[2], words[3]};
          for (int lane = 0; lane < 32; ++lane) {
            int index = start + lane;
            for (int reg = 0; reg < 3; ++reg) {
              words[reg][lane] = start < 4096 ? edge[index % 16] : random();
              index /= 16;
            }
          }
          ASSERT_EQ(goc_test::integer_mad_functions[op](cpu, UINT32_MAX,
                                                        goc_test::integer_mad_mode_bits(mode),
                                                        p + 3, p, p + 1, p + 2),
                    GOC_SUCCESS);
          for (int lane = 0; lane < 32; ++lane)
            EXPECT_EQ(words[3][lane], goc_test::integer_mad_reference(
                                          op, words[0][lane], words[1][lane], words[2][lane],
                                          goc_test::integer_mad_mode_bits(mode)));
        }
      }
}

TEST(IntegerMad, Every16BitFactorEncoding) {
  for (int op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int source = 0; source < 2; ++source)
        for (int mode = 0; mode < 8; ++mode) {
          std::mt19937 random(129);
          for (uint32_t start = 0; start < 65536; start += 32) {
            uint32_t words[4][32];
            uint32_t *p[] = {words[0], words[1], words[2], words[3]};
            for (uint32_t lane = 0; lane < 32; ++lane) {
              for (int reg = 0; reg < 3; ++reg)
                words[reg][lane] = random();
              words[source][lane] = (start + lane) | ((65535 - start - lane) << 16);
            }
            ASSERT_EQ(goc_test::integer_mad_functions[op](cpu, UINT32_MAX,
                                                          goc_test::integer_mad_mode_bits(mode),
                                                          p + 3, p, p + 1, p + 2),
                      GOC_SUCCESS);
            for (int lane = 0; lane < 32; ++lane)
              ASSERT_EQ(words[3][lane], goc_test::integer_mad_reference(
                                            op, words[0][lane], words[1][lane], words[2][lane],
                                            goc_test::integer_mad_mode_bits(mode)));
          }
        }
}

TEST(IntegerMad, DiscardedUpperBytes) {
  for (int op = 2; op < 4; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint32_t mode : {uint32_t(0), GOC_ALU_CLAMP})
        for (uint32_t start = 0; start < 65536; start += 32) {
          uint32_t words[4][32], expected[32];
          uint32_t *p[] = {words[0], words[1], words[2], words[3]};
          for (uint32_t lane = 0; lane < 32; ++lane) {
            uint32_t a = (0x71133u * (lane + 1)) & 0xffffff;
            uint32_t b = (0xfb87ddu * (lane + 3)) & 0xffffff;
            words[2][lane] = 0x47359831u * (lane + 7);
            expected[lane] = goc_test::integer_mad_reference(op, a, b, words[2][lane], mode);
            uint32_t pair = start + lane;
            words[0][lane] = a | ((pair & 255) << 24);
            words[1][lane] = b | ((pair >> 8) << 24);
          }
          ASSERT_EQ(
              goc_test::integer_mad_functions[op](cpu, UINT32_MAX, mode, p + 3, p, p + 1, p + 2),
              GOC_SUCCESS);
          for (int lane = 0; lane < 32; ++lane)
            EXPECT_EQ(words[3][lane], expected[lane]);
        }
}

TEST(IntegerMad, MasksModifiersAndAllWholeRegisterAliases) {
  for (int op = 0; op < 4; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int mode = 0; mode < (op < 2 ? 8 : 2); ++mode)
        for (uint32_t exec_mask : exec_masks())
          for (int bi = 0; bi <= 1; ++bi)
            for (int ci = 0; ci <= bi + 1; ++ci)
              for (int di = 0; di <= std::max(bi, ci) + 1; ++di) {
                uint32_t words[4][32], saved[4][32];
                uint32_t *p[] = {words[0], words[1], words[2], words[3]};
                for (int reg = 0; reg < 4; ++reg)
                  for (int lane = 0; lane < 32; ++lane)
                    words[reg][lane] = 0x7af551d3u * (lane + reg * 32 + 1);
                std::memcpy(saved, words, sizeof(words));
                ASSERT_EQ(goc_test::integer_mad_functions[op](cpu, exec_mask,
                                                              goc_test::integer_mad_mode_bits(mode),
                                                              p + di, p, p + bi, p + ci),
                          GOC_SUCCESS);
                for (int reg = 0; reg < 4; ++reg)
                  for (int lane = 0; lane < 32; ++lane) {
                    uint32_t expected =
                        reg == di && (exec_mask >> lane & 1)
                            ? goc_test::integer_mad_reference(op, saved[0][lane], saved[bi][lane],
                                                              saved[ci][lane],
                                                              goc_test::integer_mad_mode_bits(mode))
                            : saved[reg][lane];
                    ASSERT_EQ(words[reg][lane], expected)
                        << op << "/" << cpu << "/" << mode << "/" << exec_mask;
                  }
              }
}

TEST(IntegerMad, LiteralOverflowCancellationAndSelection) {
  struct Witness {
    int op;
    uint32_t a, b, c, mode, expected;
  };

  const Witness cases[] = {
      {0, 0xffff, 0xffff, 0xffffffff, 0, 0xfffe0000},
      {0, 0xffff, 0xffff, 0xffffffff, GOC_ALU_CLAMP, 0xffffffff},
      {1, 0x8000, 0x8000, 0x7fffffff, 0, 0xbfffffff},
      {1, 0x8000, 0x8000, 0x7fffffff, GOC_ALU_CLAMP, 0x7fffffff},
      {1, 0x8000, 0x7fff, 0x80000000, 0, 0x40008000},
      {1, 0x8000, 0x7fff, 0x80000000, GOC_ALU_CLAMP, 0x80000000},
      {1, 0x8000, 0x8000, 0xc0000001, GOC_ALU_CLAMP, 1},
      {3, 0x400000, 512, 0xffffffff, GOC_ALU_CLAMP, 0x7fffffff},
      {3, 0x800000, 257, 0x800000, GOC_ALU_CLAMP, 0x80000000},
      {3, 0x7fffff, 256, 256, 0, 0x80000000},
      {3, 0x7fffff, 256, 256, GOC_ALU_CLAMP, 0x7fffffff},
      {2, 0xffffff, 0xffffff, 0xffffffff, GOC_ALU_CLAMP, 0xffffffff},
      {1, 0xfffe0002, 0x00030004, 5, GOC_ALU_HIGH_A | GOC_ALU_HIGH_B, 0xffffffff},
      {0, 0x00020003, 0x00050007, 0x1234000d, GOC_ALU_HIGH_A | GOC_ALU_HIGH_B, 0x12340017},
      {2, 0xfe000002, 0x80000003, 0x40000000, 0, 0x40000006},
      {3, 0xfe000002, 0x80000003, 0x40000000, 0, 0x40000006},
  };
  for (const auto &w : cases)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t words[3][32];
      uint32_t *p[] = {words[0], words[1], words[2]};
      for (int lane = 0; lane < 32; ++lane) {
        words[0][lane] = w.a;
        words[1][lane] = w.b;
        words[2][lane] = w.c;
      }
      ASSERT_EQ(
          goc_test::integer_mad_functions[w.op](cpu, UINT32_MAX, w.mode, p + 2, p, p + 1, p + 2),
          GOC_SUCCESS);
      for (uint32_t value : words[2])
        EXPECT_EQ(value, w.expected);
    }
}

TEST(IntegerMad, ValidationAndFloatingEnvironment) {
  for (int op = 0; op < 4; ++op) {
    auto fn = goc_test::integer_mad_functions[op];
    uint32_t words[4][32] = {}, saved[32];
    uint32_t *p[] = {words[0], words[1], words[2], words[3]};
    for (int i = 0; i < 32; ++i)
      words[3][i] = 0x76543210;
    std::memcpy(saved, words[3], sizeof(saved));
    const uint32_t known = GOC_ALU_CLAMP | (op < 2 ? GOC_ALU_HIGH_A | GOC_ALU_HIGH_B : 0);
    for (int bit = 0; bit < 32; ++bit) {
      uint32_t invalid = 1U << bit;
      if (invalid & known)
        continue;
      EXPECT_EQ(fn(0, UINT32_MAX, invalid, p + 3, p, p + 1, p + 2), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(0, 0, invalid, nullptr, nullptr, nullptr, nullptr), GOC_ERROR_INVALID_FLAGS);
    }
    EXPECT_EQ(std::memcmp(saved, words[3], sizeof(saved)), 0);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0, nullptr, nullptr,
                 nullptr, nullptr),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    EXPECT_EQ(fn(1ULL << 63, 0, 0, nullptr, nullptr, nullptr, nullptr), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(fn(0, 0U, 0, nullptr, nullptr, nullptr, nullptr), GOC_SUCCESS);
    goc_test::ScopedFpEnvironment environment;
    ASSERT_TRUE(environment.saved());
    std::fesetround(FE_DOWNWARD);
    std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_INVALID);
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      EXPECT_EQ(fn(cpu | GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, GOC_ALU_CLAMP, p + 3, p, p + 1,
                   p + 2),
                GOC_SUCCESS);
      EXPECT_EQ(std::fegetround(), FE_DOWNWARD);
      EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_INVALID);
    }
  }
}

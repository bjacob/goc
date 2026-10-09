// SPDX-License-Identifier: MIT

#include "fp_environment.h"
#include "goc/goc.h"
#include "rdna4_exec_masks.h"
#include "rdna4_integer16_ternary_reference.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

TEST(Integer16Ternary, EveryModifierBoundaryTriplesAndRandomInputs) {
  const uint32_t edge[] = {0, 1, 2, 0x7fff, 0x8000, 0x8001, 0xfffe, 0xffff};
  for (int op = 0; op < 8; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int mode = 0; mode < (op < 2 ? 32 : 16); ++mode) {
        SCOPED_TRACE(::testing::Message() << op << "/" << cpu << "/" << mode);
        std::mt19937 random(995);
        for (int start = 0; start < 1536; start += 32) {
          uint32_t words[4][32];
          uint32_t *p[] = {words[0], words[1], words[2], words[3]};
          for (int lane = 0; lane < 32; ++lane) {
            words[3][lane] = 0xfacecafe;
            int index = start + lane;
            for (int reg = 0; reg < 3; ++reg) {
              words[reg][lane] =
                  start < 512 ? edge[index % 8] | (edge[7 - index % 8] << 16) : random();
              index /= 8;
            }
          }
          ASSERT_EQ(goc_test::integer16_ternary_functions[op](
                        cpu, UINT32_MAX, goc_test::integer16_ternary_mode_bits(mode), p + 3, p,
                        p + 1, p + 2),
                    GOC_SUCCESS);
          for (int lane = 0; lane < 32; ++lane)
            EXPECT_EQ(words[3][lane], goc_test::integer16_ternary_reference(
                                          op, words[0][lane], words[1][lane], words[2][lane],
                                          goc_test::integer16_ternary_mode_bits(mode)));
        }
      }
}

TEST(Integer16Ternary, EverySourceEncoding) {
  for (int op = 0; op < 8; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int source = 0; source < 3; ++source)
        for (uint32_t mode : {uint32_t(0), GOC_ALU_HIGH_A | GOC_ALU_HIGH_C | GOC_ALU_HIGH_D |
                                               (op < 2 ? GOC_ALU_CLAMP : 0)}) {
          std::mt19937 random(401);
          for (uint32_t start = 0; start < 65536; start += 32) {
            uint32_t words[4][32];
            uint32_t *p[] = {words[0], words[1], words[2], words[3]};
            for (uint32_t lane = 0; lane < 32; ++lane) {
              for (int reg = 0; reg < 3; ++reg)
                words[reg][lane] = random();
              words[source][lane] = (start + lane) | ((65535 - start - lane) << 16);
              words[3][lane] = 0xfacecafe;
            }
            ASSERT_EQ(goc_test::integer16_ternary_functions[op](cpu, UINT32_MAX, mode, p + 3, p,
                                                                p + 1, p + 2),
                      GOC_SUCCESS);
            for (int lane = 0; lane < 32; ++lane)
              ASSERT_EQ(words[3][lane],
                        goc_test::integer16_ternary_reference(op, words[0][lane], words[1][lane],
                                                              words[2][lane], mode));
          }
        }
}

TEST(Integer16Ternary, AllMasksModifiersAndWholeRegisterAliases) {
  for (int op = 0; op < 8; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int mode = 0; mode < (op < 2 ? 32 : 16); ++mode)
        for (uint32_t mask : rdna4_exec_masks())
          // Restricted growth indices enumerate all 15 partitions of A/B/C/D.
          for (int bi = 0; bi <= 1; ++bi)
            for (int ci = 0; ci <= bi + 1; ++ci)
              for (int di = 0; di <= std::max(bi, ci) + 1; ++di) {
                uint32_t words[4][32], saved[4][32];
                uint32_t *p[] = {words[0], words[1], words[2], words[3]};
                for (int reg = 0; reg < 4; ++reg)
                  for (int lane = 0; lane < 32; ++lane)
                    words[reg][lane] = 0x7af551d3u * (lane + reg * 32 + 1);
                std::memcpy(saved, words, sizeof(words));
                ASSERT_EQ(goc_test::integer16_ternary_functions[op](
                              cpu, mask, goc_test::integer16_ternary_mode_bits(mode), p + di, p,
                              p + bi, p + ci),
                          GOC_SUCCESS);
                for (int reg = 0; reg < 4; ++reg)
                  for (int lane = 0; lane < 32; ++lane) {
                    uint32_t expected =
                        reg == di && (mask >> lane & 1)
                            ? goc_test::integer16_ternary_reference(
                                  op, saved[0][lane], saved[bi][lane], saved[ci][lane],
                                  goc_test::integer16_ternary_mode_bits(mode), saved[di][lane])
                            : saved[reg][lane];
                    ASSERT_EQ(words[reg][lane], expected)
                        << op << "/" << cpu << "/" << mode << "/" << mask;
                  }
              }
}

TEST(Integer16Ternary, LiteralSaturationSelectionAndPreservedHalf) {
  struct Witness {
    int op;
    uint32_t a, b, c, mode, expected;
  };

  const Witness cases[] = {
      {6, 0x12348000, 0xabcd7fff, 0x4321ffff, 0, 0x12348000},
      {7, 0x12348000, 0xabcd7fff, 0x4321ffff, GOC_ALU_HIGH_D, 0xffff8000},
      {7, 0x0002ffff, 0xffff0002, 0x7fff8000, GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_C,
       0x00020002},
      {0, 0x1234ffff, 0x5678ffff, 0x9abcffff, 0, 0x12340000},
      {0, 0x1234ffff, 0x5678ffff, 0x9abcffff, GOC_ALU_CLAMP | GOC_ALU_HIGH_D, 0xffffffff},
      {1, 0x80007fff, 0x00020002, 0x7fff8000, GOC_ALU_CLAMP, 0x80007ffe},
      {1, 0x80007fff, 0x00020002, 0x7fff8000,
       GOC_ALU_CLAMP | GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_C | GOC_ALU_HIGH_D,
       0x80007fff},
      {1, 0x12348000, 0x43218000, 0xaaaa8000, GOC_ALU_CLAMP, 0x12347fff},
      {0, 0x00020003, 0x00050007, 0x000b000d, GOC_ALU_HIGH_A | GOC_ALU_HIGH_C | GOC_ALU_HIGH_D,
       0x00190003},
      {2, 0x8000ffff, 0x7fff1234, 0x00010002, GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_C,
       0x80000001},
      {3, 0x8000ffff, 0x7fff1234, 0x00010002, GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_C,
       0x80008000},
      {4, 0x8000ffff, 0x7fff1234, 0x00010002,
       GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_C | GOC_ALU_HIGH_D, 0x8000ffff},
      {5, 0x8000ffff, 0x7fff1234, 0x00010002,
       GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_C | GOC_ALU_HIGH_D, 0x7fffffff},
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
          goc_test::integer16_ternary_functions[w.op](cpu, UINT32_MAX, w.mode, p, p, p + 1, p + 2),
          GOC_SUCCESS);
      for (uint32_t value : words[0])
        EXPECT_EQ(value, w.expected);
    }
}

TEST(Integer16Ternary, ValidationAndFloatingEnvironment) {
  for (int op = 0; op < 8; ++op) {
    auto fn = goc_test::integer16_ternary_functions[op];
    uint32_t words[4][32] = {}, saved[32];
    uint32_t *p[] = {words[0], words[1], words[2], words[3]};
    for (int i = 0; i < 32; ++i)
      words[3][i] = 0x76543210;
    std::memcpy(saved, words[3], sizeof(saved));
    const uint32_t known = GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_C | GOC_ALU_HIGH_D |
                           (op < 2 ? GOC_ALU_CLAMP : 0);
    for (int bit = 0; bit < 32; ++bit) {
      uint32_t invalid = UINT32_C(1) << bit;
      if (invalid & known)
        continue;
      EXPECT_EQ(fn(0, UINT32_MAX, invalid, p + 3, p, p + 1, p + 2), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(0, 0, invalid, nullptr, nullptr, nullptr, nullptr), GOC_ERROR_INVALID_FLAGS);
    }
    EXPECT_EQ(std::memcmp(saved, words[3], sizeof(saved)), 0);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0, nullptr, nullptr,
                 nullptr, nullptr),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    EXPECT_EQ(fn(UINT64_C(1) << 63, 0, 0, nullptr, nullptr, nullptr, nullptr),
              GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(fn(0, UINT32_C(0), 0, nullptr, nullptr, nullptr, nullptr), GOC_SUCCESS);
    goc_test::ScopedFpEnvironment environment;
    ASSERT_TRUE(environment.saved());
    std::fesetround(FE_UPWARD);
    std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_DIVBYZERO);
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      EXPECT_EQ(fn(cpu | GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, GOC_ALU_HIGH_C | GOC_ALU_HIGH_D,
                   p + 3, p, p + 1, p + 2),
                GOC_SUCCESS);
      EXPECT_EQ(std::fegetround(), FE_UPWARD);
      EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
    }
  }
}

// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_exec_masks.h"
#include "rdna4_integer16_reference.h"

#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

TEST(Integer16, EveryModifierBoundaryPairsAndRandomInputs) {
  const uint32_t values[] = {0, 1, 2, 0x7ffe, 0x7fff, 0x8000, 0x8001, 0xfffe, 0xffff};
  for (int op = 0; op < 12; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int mode = 0; mode < (op < 4 ? 16 : 8); ++mode) {
        SCOPED_TRACE(::testing::Message() << op << "/" << cpu << "/" << mode);
        std::mt19937 random(312);
        for (int start = 0; start < 1152; start += 32) {
          uint32_t a[32], b[32], d[32];
          const uint32_t *ap[] = {a}, *bp[] = {b};
          uint32_t *dp[] = {d};
          for (int lane = 0; lane < 32; ++lane) {
            int i = start + lane;
            a[lane] = i < 128 ? values[i % 9] | (values[(i / 9) % 9] << 16) : random();
            b[lane] = i < 128 ? values[(i / 9) % 9] | (values[8 - i % 9] << 16) : random();
            d[lane] = 0xfacecafe;
          }
          ASSERT_EQ(goc_test::integer16_functions[op](
                        cpu, UINT32_MAX, goc_test::integer16_mode_bits(mode), dp, ap, bp),
                    GOC_SUCCESS);
          for (int lane = 0; lane < 32; ++lane)
            EXPECT_EQ(d[lane], goc_test::integer16_reference(op, a[lane], b[lane], 0xfacecafe,
                                                             goc_test::integer16_mode_bits(mode)));
        }
      }
}

TEST(Integer16, EveryInputEncoding) {
  for (int op = 0; op < 12; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int source = 0; source < 2; ++source)
        for (uint32_t mode : {uint32_t(0), GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_D |
                                               (op < 4 ? GOC_ALU_CLAMP : 0)}) {
          std::mt19937 random(597);
          for (uint32_t start = 0; start < 65536; start += 32) {
            uint32_t words[3][32];
            uint32_t *p[] = {words[0], words[1], words[2]};
            for (uint32_t lane = 0; lane < 32; ++lane) {
              words[0][lane] = random();
              words[1][lane] = random();
              words[source][lane] = (start + lane) | ((65535 - start - lane) << 16);
              words[2][lane] = 0x1234abcd;
            }
            ASSERT_EQ(goc_test::integer16_functions[op](cpu, UINT32_MAX, mode, p + 2, p, p + 1),
                      GOC_SUCCESS);
            for (int lane = 0; lane < 32; ++lane)
              ASSERT_EQ(words[2][lane], goc_test::integer16_reference(
                                            op, words[0][lane], words[1][lane], 0x1234abcd, mode));
          }
        }
}

TEST(Integer16, MasksModifiersAndAllWholeRegisterAliases) {
  const int layouts[][3] = {{0, 1, 2}, {0, 0, 2}, {0, 1, 0}, {0, 1, 1}, {0, 0, 0}};
  for (int op = 0; op < 12; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int mode = 0; mode < (op < 4 ? 16 : 8); ++mode)
        for (uint64_t mask : rdna4_exec_masks())
          for (const auto &layout : layouts) {
            uint32_t words[3][32], saved[3][32];
            for (int r = 0; r < 3; ++r)
              for (int lane = 0; lane < 32; ++lane)
                words[r][lane] = 0x913b7a25u * (lane + 17 * r + 1);
            std::memcpy(saved, words, sizeof(words));
            const uint32_t *a[] = {words[layout[0]]}, *b[] = {words[layout[1]]};
            uint32_t *d[] = {words[layout[2]]};
            ASSERT_EQ(goc_test::integer16_functions[op](
                          cpu, mask, goc_test::integer16_mode_bits(mode), d, a, b),
                      GOC_SUCCESS);
            for (int r = 0; r < 3; ++r)
              for (int lane = 0; lane < 32; ++lane) {
                uint32_t expected = r == layout[2] && (mask >> lane & 1)
                                        ? goc_test::integer16_reference(
                                              op, saved[layout[0]][lane], saved[layout[1]][lane],
                                              saved[r][lane], goc_test::integer16_mode_bits(mode))
                                        : saved[r][lane];
                ASSERT_EQ(words[r][lane], expected)
                    << op << "/" << cpu << "/" << mode << "/" << mask;
              }
          }
}

TEST(Integer16, LiteralSelectionSaturationAndShiftWitnesses) {
  struct Witness {
    int op;
    uint32_t a, b, mode, expected;
  };

  const Witness cases[] = {
      {0, 0x80007fff, 0xffff0001, 0, 0x80008000},
      {0, 0x80007fff, 0xffff0001, GOC_ALU_CLAMP, 0x80007fff},
      {0, 0x80007fff, 0xffff0001, GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_D | GOC_ALU_CLAMP,
       0x80007fff},
      {1, 0x7fff8000, 0xffff0001, GOC_ALU_CLAMP, 0x7fff8000},
      {2, 0x80007fff, 0xffff0001, GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_CLAMP, 0x8000ffff},
      {3, 0x12340000, 0xabcd0001, GOC_ALU_CLAMP, 0x12340000},
      {4, 0x7fff8000, 0x80007fff, GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_D, 0x80008000},
      {6, 0x7fff8000, 0x80007fff, GOC_ALU_HIGH_D, 0x7fff8000},
      {8, 0xffffffff, 0xffffffff, GOC_ALU_HIGH_D, 0x0001ffff},
      {9, 0xfff0000f, 0xaaaa0001, 0, 0xfff08000},
      {10, 0xbeef0001, 0x12348000, 0, 0xbeef4000},
      {11, 0xbeef0001, 0x12348000, GOC_ALU_HIGH_D, 0xc0000001},
      {11, 0x000ffff0, 0x80017fff, GOC_ALU_HIGH_A | GOC_ALU_HIGH_B, 0x000fffff},
  };
  for (const auto &w : cases)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t a[32], b[32];
      for (int lane = 0; lane < 32; ++lane) {
        a[lane] = w.a;
        b[lane] = w.b;
      }
      const uint32_t *ap[] = {a}, *bp[] = {b};
      uint32_t *dp[] = {a};
      ASSERT_EQ(goc_test::integer16_functions[w.op](cpu, UINT32_MAX, w.mode, dp, ap, bp),
                GOC_SUCCESS);
      for (uint32_t value : a)
        EXPECT_EQ(value, w.expected);
    }
}

TEST(Integer16, ValidationAndFloatingEnvironment) {
  for (int op = 0; op < 12; ++op) {
    auto fn = goc_test::integer16_functions[op];
    uint32_t a[32] = {}, b[32] = {}, d[32], saved[32];
    for (int i = 0; i < 32; ++i)
      d[i] = 0x12345678;
    std::memcpy(saved, d, sizeof(d));
    const uint32_t *ap[] = {a}, *bp[] = {b};
    uint32_t *dp[] = {d};
    const uint32_t known =
        GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_D | (op < 4 ? GOC_ALU_CLAMP : 0);
    for (int bit = 0; bit < 32; ++bit) {
      uint32_t invalid = UINT32_C(1) << bit;
      if (invalid & known)
        continue;
      EXPECT_EQ(fn(0, UINT32_MAX, invalid, dp, ap, bp), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(0, 0, invalid, nullptr, nullptr, nullptr), GOC_ERROR_INVALID_FLAGS);
    }
    EXPECT_EQ(std::memcmp(saved, d, sizeof(d)), 0);
    EXPECT_EQ(
        fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0, nullptr, nullptr, nullptr),
        GOC_ERROR_UNSUPPORTED_SEMANTICS);
    EXPECT_EQ(fn(0, UINT64_C(0xffffffff00000000), 0, nullptr, nullptr, nullptr), GOC_SUCCESS);
    fenv_t environment;
    ASSERT_EQ(std::fegetenv(&environment), 0);
    std::fesetround(FE_DOWNWARD);
    std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_INVALID);
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      EXPECT_EQ(fn(cpu | GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, GOC_ALU_HIGH_D, dp, ap, bp),
                GOC_SUCCESS);
      EXPECT_EQ(std::fegetround(), FE_DOWNWARD);
      EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_INVALID);
    }
    std::fesetenv(&environment);
  }
}

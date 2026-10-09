// SPDX-License-Identifier: MIT

#include "fp_environment.h"
#include "goc/goc.h"
#include "rdna4_exec_masks.h"
#include "rdna4_integer_mul_reference.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <random>
#include <stdint.h>

TEST(IntegerMul, BoundaryProductsAndRandomInputs) {
  const uint32_t boundary[] = {0,          1,          2,          46340,      46341,
                               65535,      65536,      0x007ffffe, 0x007fffff, 0x00800000,
                               0x00800001, 0x00ffffff, 0x01000000, 0x01000001, 0x7fffffff,
                               0x80000000, 0x80000001, 0xffffffff, 0xff7fffff, 0xff800000};
  for (int op = 0; op < 7; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int clamp = 0; clamp <= int(goc_test::integer_mul_can_clamp(op)); ++clamp) {
        std::mt19937 random(371);
        for (int start = 0; start < 400 + 1024; start += 32) {
          SCOPED_TRACE(::testing::Message() << op << "/" << cpu << "/" << clamp << "/" << start);
          uint32_t a[32], b[32], d[32];
          auto pa = a, pb = b, pd = d;
          for (int lane = 0; lane < 32; ++lane) {
            int i = start + lane;
            a[lane] = i < 400 ? boundary[i % 20] : random();
            b[lane] = i < 400 ? boundary[i / 20] : random();
          }
          ASSERT_EQ(goc_test::integer_mul_functions[op](cpu, UINT32_MAX, clamp ? GOC_ALU_CLAMP : 0,
                                                        &pd, &pa, &pb),
                    GOC_SUCCESS);
          for (int lane = 0; lane < 32; ++lane)
            EXPECT_EQ(d[lane], goc_test::integer_mul_reference(op, a[lane], b[lane], clamp));
        }
      }
}

TEST(IntegerMul, MasksAliasesAndSaturation) {
  const int layouts[][2] = {{0, 1}, {0, 0}, {1, 0}};
  for (int op = 0; op < 7; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int clamp = 0; clamp <= int(goc_test::integer_mul_can_clamp(op)); ++clamp)
        for (uint32_t mask : rdna4_exec_masks())
          for (const auto &layout : layouts)
            for (int dest = 0; dest < 3; ++dest) {
              SCOPED_TRACE(::testing::Message() << op << "/" << cpu << "/" << clamp << "/" << mask
                                                << "/" << layout[0] << layout[1] << "/" << dest);
              uint32_t words[3][34], want[3][34];
              std::mt19937 random(719);
              for (auto &reg : words) {
                std::fill(reg, reg + 34, 0xdeadbeef);
                for (int lane = 1; lane <= 32; ++lane)
                  reg[lane] = random();
              }
              std::memcpy(want, words, sizeof(want));
              for (int lane = 0; lane < 32; ++lane)
                if (mask >> lane & 1)
                  want[dest][lane + 1] = goc_test::integer_mul_reference(
                      op, words[layout[0]][lane + 1], words[layout[1]][lane + 1], clamp);
              uint32_t *a = words[layout[0]] + 1, *b = words[layout[1]] + 1, *d = words[dest] + 1;
              ASSERT_EQ(goc_test::integer_mul_functions[op](cpu, mask, clamp ? GOC_ALU_CLAMP : 0,
                                                            &d, &a, &b),
                        GOC_SUCCESS);
              for (int reg = 0; reg < 3; ++reg)
                for (int lane = 0; lane < 34; ++lane)
                  EXPECT_EQ(words[reg][lane], want[reg][lane]);
            }
}

TEST(IntegerMul, LiteralHighWordsSignExtensionAndClamp) {
  struct Case {
    int op;
    uint32_t a, b, mode, expected;
  };

  const Case cases[] = {{0, 0xffffffff, 0xffffffff, 0, 1},
                        {1, 0xffffffff, 0xffffffff, 0, 0xfffffffe},
                        {2, 0xffffffff, 2, 0, 0xffffffff},
                        {2, 0x80000000, 0x80000000, 0, 0x40000000},
                        {3, 0x00ffffff, 2, 0, 0xfffffffe},
                        {3, 0xff000001, 2, 0, 2},
                        {3, 0x00800000, 0x00800000, 0, 0},
                        {3, 0x00800000, 0x00800000, GOC_ALU_CLAMP, 0x7fffffff},
                        {3, 0x00800000, 0x007fffff, GOC_ALU_CLAMP, 0x80000000},
                        {3, 0x00800000, 256, GOC_ALU_CLAMP, 0x80000000},
                        {4, 0x00ffffff, 2, 0, 0xffffffff},
                        {4, 0x00800000, 0x00800000, 0, 0x00004000},
                        {5, 0x00ffffff, 0x00ffffff, 0, 0xfe000001},
                        {5, 0x00ffffff, 0x00ffffff, GOC_ALU_CLAMP, 0xffffffff},
                        {5, 0xff000001, 2, GOC_ALU_CLAMP, 2},
                        {6, 0x00ffffff, 0x00ffffff, 0, 0x0000ffff}};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (const auto &test : cases) {
      uint32_t a[32], b[32], d[32];
      auto pa = a, pb = b, pd = d;
      std::fill(a, a + 32, test.a);
      std::fill(b, b + 32, test.b);
      ASSERT_EQ(goc_test::integer_mul_functions[test.op](cpu, UINT32_MAX, test.mode, &pd, &pa, &pb),
                GOC_SUCCESS);
      for (uint32_t value : d)
        EXPECT_EQ(value, test.expected);
    }
}

TEST(IntegerMul, ValidationAndFpEnvironment) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (int op = 0; op < 7; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t a[32], b[32], d[32];
      auto pa = a, pb = b, pd = d;
      std::fill(a, a + 32, 0x7f800001);
      std::fill(b, b + 32, 0xffffffff);
      std::fill(d, d + 32, 0xdeadbeef);
      for (uint32_t mask : {0U, UINT32_MAX}) {
        for (int bit = 0; bit < 32; ++bit) {
          uint32_t mode = 1U << bit;
          if (mode == GOC_ALU_CLAMP && goc_test::integer_mul_can_clamp(op))
            continue;
          EXPECT_EQ(goc_test::integer_mul_functions[op](cpu, mask, mode, &pd, &pa, &pb),
                    GOC_ERROR_INVALID_FLAGS);
        }
        EXPECT_EQ(goc_test::integer_mul_functions[op](cpu | (1ULL << 63), mask, 0, &pd, &pa, &pb),
                  GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(goc_test::integer_mul_functions[op](cpu | GOC_SEMANTICS_EXACT_EMPIRICAL |
                                                          GOC_SEMANTICS_STRICT,
                                                      mask, 0, &pd, &pa, &pb),
                  GOC_ERROR_UNSUPPORTED_SEMANTICS);
      }
      for (auto value : d)
        EXPECT_EQ(value, 0xdeadbeef);
      for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        std::fesetround(rounding);
        std::feclearexcept(FE_ALL_EXCEPT);
        std::feraiseexcept(FE_DIVBYZERO);
        EXPECT_EQ(goc_test::integer_mul_functions[op](
                      cpu | GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX,
                      goc_test::integer_mul_can_clamp(op) ? GOC_ALU_CLAMP : 0, &pd, &pa, &pb),
                  GOC_SUCCESS);
        EXPECT_EQ(std::fegetround(), rounding);
        EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
      }
    }
}

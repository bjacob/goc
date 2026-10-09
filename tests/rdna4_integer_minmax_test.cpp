// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_exec_masks.h"
#include "rdna4_integer_minmax_reference.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <random>
#include <stdint.h>

TEST(IntegerMinmax, BoundaryCartesianProductsAndRandomValues) {
  const uint32_t boundary[] = {0,          1,          2,          0x7ffffffe,
                               0x7fffffff, 0x80000000, 0x80000001, 0xfffffffe,
                               0xffffffff, 0x7f800001, 0xff800001, 0x3f800000};
  for (int op = 0; op < 14; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      std::mt19937 random(307);
      for (int start = 0; start < 1728 + 1024; start += 32) {
        SCOPED_TRACE(::testing::Message() << op << "/" << cpu << "/" << start);
        uint32_t words[4][32];
        uint32_t *p[] = {words[0], words[1], words[2], words[3]};
        for (int lane = 0; lane < 32; ++lane) {
          int index = start + lane;
          for (int reg = 0; reg < 3; ++reg) {
            words[reg][lane] = start < 1728 ? boundary[index % 12] : random();
            index /= 12;
          }
        }
        ASSERT_EQ(
            goc_test::integer_minmax_functions[op](cpu, UINT32_MAX, 0, p + 3, p, p + 1, p + 2),
            GOC_SUCCESS);
        for (int lane = 0; lane < 32; ++lane)
          EXPECT_EQ(words[3][lane], goc_test::integer_minmax_reference(
                                        op, words[0][lane], words[1][lane], words[2][lane]));
      }
    }
}

TEST(IntegerMinmax, MasksAndAllWholeRegisterAliases) {
  const int layouts[][3] = {{0, 1, 2}, {0, 0, 2}, {0, 1, 0}, {0, 1, 1}, {0, 0, 0}};
  for (int op = 0; op < 14; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint32_t mask : rdna4_exec_masks())
        for (const auto &layout : layouts)
          for (int dest = 0; dest < 4; ++dest) {
            SCOPED_TRACE(::testing::Message() << op << "/" << cpu << "/" << mask << "/" << dest
                                              << "/" << layout[0] << layout[1] << layout[2]);
            std::mt19937 random(109);
            uint32_t words[4][34], want[4][34];
            for (auto &reg : words) {
              std::fill(reg, reg + 34, 0xdeadbeef);
              for (int lane = 1; lane <= 32; ++lane)
                reg[lane] = random();
            }
            std::memcpy(want, words, sizeof(want));
            for (int lane = 0; lane < 32; ++lane)
              if (mask >> lane & 1)
                want[dest][lane + 1] = goc_test::integer_minmax_reference(
                    op, words[layout[0]][lane + 1], words[layout[1]][lane + 1],
                    words[layout[2]][lane + 1]);
            uint32_t *d = words[dest] + 1, *a = words[layout[0]] + 1, *b = words[layout[1]] + 1,
                     *c = words[layout[2]] + 1;
            ASSERT_EQ(goc_test::integer_minmax_functions[op](cpu, mask, 0, &d, &a, &b, &c),
                      GOC_SUCCESS);
            for (int reg = 0; reg < 4; ++reg)
              for (int lane = 0; lane < 34; ++lane)
                EXPECT_EQ(words[reg][lane], want[reg][lane]);
          }
}

TEST(IntegerMinmax, LiteralSignednessAndOperandOrder) {
  const uint32_t expected[] = {1, 2, 1, 3, 3, 2, 2, 1, 2, 1, 3, 3, 2, 2};
  const uint32_t signs[] = {0xffffffff, 1,          0x80000000, 1,          0xffffffff,
                            0x80000000, 0xffffffff, 1,          0xffffffff, 1,
                            0xffffffff, 0x80000000, 0x80000000, 0x80000000};
  for (int op = 0; op < 14; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t words[4][32];
      uint32_t *p[] = {words[0], words[1], words[2], words[3]};
      for (int lane = 0; lane < 32; ++lane) {
        words[0][lane] = lane & 1 ? 0xffffffff : 1;
        words[1][lane] = lane & 1 ? 1 : 2;
        words[2][lane] = lane & 1 ? 0x80000000 : 3;
      }
      ASSERT_EQ(goc_test::integer_minmax_functions[op](cpu, UINT32_MAX, 0, p + 3, p, p + 1, p + 2),
                GOC_SUCCESS);
      for (int lane = 0; lane < 32; ++lane)
        EXPECT_EQ(words[3][lane], lane & 1 ? signs[op] : expected[op]);
    }
}

TEST(IntegerMinmax, ValidationAndFpEnvironment) {
  std::fenv_t saved;
  std::fegetenv(&saved);
  for (auto fn : goc_test::integer_minmax_functions)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t input[32], output[32];
      std::fill(input, input + 32, 0x7f800001);
      std::fill(output, output + 32, 0xdeadbeef);
      auto a = input, d = output;
      for (uint32_t mask : {UINT32_C(0), UINT32_MAX}) {
        for (int bit = 0; bit < 32; ++bit)
          EXPECT_EQ(fn(cpu, mask, UINT32_C(1) << bit, &d, &a, &a, &a), GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(fn(cpu | (UINT64_C(1) << 63), mask, 0, &d, &a, &a, &a), GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(
            fn(cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, mask, 0, &d, &a, &a, &a),
            GOC_ERROR_UNSUPPORTED_SEMANTICS);
      }
      for (auto word : output)
        EXPECT_EQ(word, 0xdeadbeef);
      for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        std::fesetround(rounding);
        std::feclearexcept(FE_ALL_EXCEPT);
        std::feraiseexcept(FE_DIVBYZERO);
        EXPECT_EQ(fn(cpu | GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, 0, &d, &a, &a, &a),
                  GOC_SUCCESS);
        EXPECT_EQ(std::fegetround(), rounding);
        EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
      }
    }
  std::fesetenv(&saved);
}

TEST(IntegerMinmax, EverySingleLaneAndComplementWithAlias) {
  for (unsigned op = 0; op < 14; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned bit = 0; bit < 32; ++bit)
        for (uint32_t mask : {UINT32_C(1) << bit, ~(UINT32_C(1) << bit)}) {
          uint32_t words[3][32], expected[32];
          for (unsigned lane = 0; lane < 32; ++lane) {
            words[0][lane] = UINT32_C(0x80000000) + lane;
            words[1][lane] = UINT32_C(0x7fffffff) - lane;
            words[2][lane] = lane & 1 ? UINT32_MAX : 0;
            expected[lane] = mask & (UINT32_C(1) << lane)
                                 ? goc_test::integer_minmax_reference(
                                       op, words[0][lane], words[1][lane], words[2][lane])
                                 : words[1][lane];
          }
          const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
          uint32_t *d[] = {words[1]};
          ASSERT_EQ(goc_test::integer_minmax_functions[op](cpu, mask, 0, d, a, b,
                                                           op % 7 < 2 ? nullptr : c),
                    GOC_SUCCESS);
          for (unsigned lane = 0; lane < 32; ++lane)
            EXPECT_EQ(words[1][lane], expected[lane]);
        }
}

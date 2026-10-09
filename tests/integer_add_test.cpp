// SPDX-License-Identifier: MIT

#include "exec_masks.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "integer_add_reference.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <random>
#include <stdint.h>

TEST(IntegerAdd, BoundaryCartesianProductsAndRandomInputs) {
  const uint32_t values[] = {0,          1,          2,          0x7ffffffe,
                             0x7fffffff, 0x80000000, 0x80000001, 0xfffffffe,
                             0xffffffff, 0x7f800001, 0xff800001, 0x40000000};
  for (int op = 0; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int clamp = 0; clamp <= int(op != 5); ++clamp) {
        std::mt19937 random(421);
        for (int start = 0; start < 1728 + 1024; start += 32) {
          SCOPED_TRACE(::testing::Message() << op << "/" << cpu << "/" << clamp << "/" << start);
          uint32_t words[4][32];
          uint32_t *p[] = {words[0], words[1], words[2], words[3]};
          for (int lane = 0; lane < 32; ++lane) {
            int i = start + lane;
            for (int reg = 0; reg < 3; ++reg) {
              words[reg][lane] = start < 1728 ? values[i % 12] : random();
              i /= 12;
            }
          }
          ASSERT_EQ(goc_test::integer_add_functions[op](cpu, UINT32_MAX, clamp ? GOC_ALU_CLAMP : 0,
                                                        p + 3, p, p + 1, p + 2),
                    GOC_SUCCESS);
          for (int lane = 0; lane < 32; ++lane)
            EXPECT_EQ(words[3][lane],
                      goc_test::integer_add_reference(op, words[0][lane], words[1][lane],
                                                      words[2][lane], clamp));
        }
      }
}

TEST(IntegerAdd, MasksAliasesAndSaturation) {
  const int layouts[][3] = {{0, 1, 2}, {0, 0, 2}, {0, 1, 0}, {0, 1, 1}, {0, 0, 0}};
  for (int op = 0; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int clamp = 0; clamp <= int(op != 5); ++clamp)
        for (uint32_t exec_mask : exec_masks())
          for (const auto &layout : layouts)
            for (int dest = 0; dest < 4; ++dest) {
              SCOPED_TRACE(::testing::Message()
                           << op << "/" << cpu << "/" << clamp << "/" << exec_mask << "/" << dest
                           << "/" << layout[0] << layout[1] << layout[2]);
              uint32_t words[4][34], want[4][34];
              std::mt19937 random(913);
              for (auto &reg : words) {
                std::fill(reg, reg + 34, 0xdeadbeef);
                for (int lane = 1; lane <= 32; ++lane)
                  reg[lane] = random();
              }
              std::memcpy(want, words, sizeof(want));
              for (int lane = 0; lane < 32; ++lane)
                if (exec_mask >> lane & 1)
                  want[dest][lane + 1] = goc_test::integer_add_reference(
                      op, words[layout[0]][lane + 1], words[layout[1]][lane + 1],
                      words[layout[2]][lane + 1], clamp);
              uint32_t *a = words[layout[0]] + 1, *b = words[layout[1]] + 1,
                       *c = words[layout[2]] + 1, *d = words[dest] + 1;
              ASSERT_EQ(goc_test::integer_add_functions[op](
                            cpu, exec_mask, clamp ? GOC_ALU_CLAMP : 0, &d, &a, &b, &c),
                        GOC_SUCCESS);
              for (int reg = 0; reg < 4; ++reg)
                for (int lane = 0; lane < 34; ++lane)
                  EXPECT_EQ(words[reg][lane], want[reg][lane]);
            }
}

TEST(IntegerAdd, LiteralOverflowBorrowAndOperandOrder) {
  struct Case {
    int op;
    uint32_t a, b, c, wrap, clamp;
  };

  const Case cases[] = {{0, 0xffffffff, 1, 0, 0, 0xffffffff},
                        {0, 0xffffffff, 0xffffffff, 0, 0xfffffffe, 0xffffffff},
                        {1, 0, 1, 0, 0xffffffff, 0},
                        {1, 0xffffffff, 1, 0, 0xfffffffe, 0xfffffffe},
                        {2, 1, 0, 0, 0xffffffff, 0},
                        {2, 1, 0xffffffff, 0, 0xfffffffe, 0xfffffffe},
                        {3, 0x7fffffff, 1, 0, 0x80000000, 0x7fffffff},
                        {3, 0x80000000, 0xffffffff, 0, 0x7fffffff, 0x80000000},
                        {3, 0x80000000, 0x80000000, 0, 0, 0x80000000},
                        {3, 0x80000000, 0x7fffffff, 0, 0xffffffff, 0xffffffff},
                        {4, 0x80000000, 1, 0, 0x7fffffff, 0x80000000},
                        {4, 0x7fffffff, 0xffffffff, 0, 0x80000000, 0x7fffffff},
                        {4, 0xffffffff, 0x80000000, 0, 0x7fffffff, 0x7fffffff},
                        {5, 0xffffffff, 0xffffffff, 0xffffffff, 0xfffffffd, 0},
                        {5, 0x80000000, 0x80000000, 1, 1, 0},
                        {5, 1, 2, 3, 6, 0}};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (const auto &test : cases)
      for (int clamp = 0; clamp <= int(test.op != 5); ++clamp) {
        uint32_t words[4][32];
        uint32_t *p[] = {words[0], words[1], words[2], words[3]};
        std::fill(words[0], words[0] + 32, test.a);
        std::fill(words[1], words[1] + 32, test.b);
        std::fill(words[2], words[2] + 32, test.c);
        ASSERT_EQ(goc_test::integer_add_functions[test.op](
                      cpu, UINT32_MAX, clamp ? GOC_ALU_CLAMP : 0, p + 3, p, p + 1, p + 2),
                  GOC_SUCCESS);
        for (uint32_t word : words[3])
          EXPECT_EQ(word, clamp ? test.clamp : test.wrap);
      }
}

TEST(IntegerAdd, ValidationAndFpEnvironment) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (int op = 0; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t input[32], output[32];
      std::fill(input, input + 32, 0x7f800001);
      std::fill(output, output + 32, 0xdeadbeef);
      auto a = input, d = output;
      for (uint32_t exec_mask : {0U, UINT32_MAX}) {
        for (int bit = 0; bit < 32; ++bit) {
          uint32_t mode = 1U << bit;
          if (op != 5 && mode == GOC_ALU_CLAMP)
            continue;
          EXPECT_EQ(goc_test::integer_add_functions[op](cpu, exec_mask, mode, &d, &a, &a, &a),
                    GOC_ERROR_INVALID_FLAGS);
        }
        EXPECT_EQ(
            goc_test::integer_add_functions[op](cpu | (1ULL << 63), exec_mask, 0, &d, &a, &a, &a),
            GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(goc_test::integer_add_functions[op](cpu | GOC_SEMANTICS_EXACT_EMPIRICAL |
                                                          GOC_SEMANTICS_STRICT,
                                                      exec_mask, 0, &d, &a, &a, &a),
                  GOC_ERROR_UNSUPPORTED_SEMANTICS);
      }
      for (uint32_t word : output)
        EXPECT_EQ(word, 0xdeadbeef);
      for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        std::fesetround(rounding);
        std::feclearexcept(FE_ALL_EXCEPT);
        std::feraiseexcept(FE_DIVBYZERO);
        EXPECT_EQ(goc_test::integer_add_functions[op](cpu | GOC_SEMANTICS_EXACT_EMPIRICAL,
                                                      UINT32_MAX, op == 5 ? 0 : GOC_ALU_CLAMP, &d,
                                                      &a, &a, &a),
                  GOC_SUCCESS);
        EXPECT_EQ(std::fegetround(), rounding);
        EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
      }
    }
}

TEST(IntegerAdd, SignedSaturationMixedLaneSigns) {
  const uint32_t a_values[] = {0x7fffffff, 0x80000000, 0x7fffffff, 0x80000000,
                               0x40000000, 0xc0000000, 0,          0xffffffff};
  const uint32_t b_values[] = {1, 0xffffffff, 0xffffffff, 1, 0x40000000, 0xc0000001, 0x80000000, 1};
  const uint32_t expected[] = {0x7fffffff, 0x80000000, 0x7ffffffe, 0x80000001,
                               0x7fffffff, 0x80000001, 0x80000000, 0};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (uint32_t exec_mask : exec_masks()) {
      uint32_t a[32], b[32];
      for (unsigned lane = 0; lane < 32; ++lane) {
        a[lane] = a_values[lane % 8];
        b[lane] = b_values[lane % 8];
      }
      // Adjacent lanes need opposite saturation limits; D also aliases A.
      uint32_t *d = a;
      const uint32_t *ap = a, *bp = b;
      ASSERT_EQ(goc_v_add_nc_i32(cpu, exec_mask, GOC_ALU_CLAMP, &d, &ap, &bp), GOC_SUCCESS);
      for (unsigned lane = 0; lane < 32; ++lane)
        EXPECT_EQ(a[lane], (exec_mask >> lane & 1) ? expected[lane % 8] : a_values[lane % 8]);
    }
}

// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_exec_masks.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <random>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_rdna4_v_add3_u32);

template <auto Function>
int binary(uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
           const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *) {
  return Function(flags, mask, mode, d, a, b);
}

const Fn functions[] = {binary<goc_rdna4_v_add_nc_u32>,    binary<goc_rdna4_v_sub_nc_u32>,
                        binary<goc_rdna4_v_subrev_nc_u32>, binary<goc_rdna4_v_add_nc_i32>,
                        binary<goc_rdna4_v_sub_nc_i32>,    goc_rdna4_v_add3_u32};

uint32_t reference(int op, uint32_t a, uint32_t b, uint32_t c, bool clamp) {
  int64_t x = a, y = b;
  bool signed_op = op == 3 || op == 4;
  if (signed_op) {
    if (x > INT32_MAX)
      x -= INT64_C(4294967296);
    if (y > INT32_MAX)
      y -= INT64_C(4294967296);
  }
  int64_t result = op == 1 || op == 4 ? x - y : op == 2 ? y - x : x + y;
  if (op == 5)
    result += c;
  if (clamp) {
    int64_t low = signed_op ? INT32_MIN : 0, high = signed_op ? INT32_MAX : int64_t(UINT32_MAX);
    if (result < low)
      result = low;
    if (result > high)
      result = high;
  }
  return uint32_t(result);
}

} // namespace

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
          ASSERT_EQ(
              functions[op](cpu, UINT32_MAX, clamp ? GOC_ALU_CLAMP : 0, p + 3, p, p + 1, p + 2),
              GOC_SUCCESS);
          for (int lane = 0; lane < 32; ++lane)
            EXPECT_EQ(words[3][lane],
                      reference(op, words[0][lane], words[1][lane], words[2][lane], clamp));
        }
      }
}

TEST(IntegerAdd, MasksAliasesAndSaturation) {
  const int layouts[][3] = {{0, 1, 2}, {0, 0, 2}, {0, 1, 0}, {0, 1, 1}, {0, 0, 0}};
  for (int op = 0; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int clamp = 0; clamp <= int(op != 5); ++clamp)
        for (uint64_t mask : rdna4_exec_masks())
          for (const auto &layout : layouts)
            for (int dest = 0; dest < 4; ++dest) {
              SCOPED_TRACE(::testing::Message()
                           << op << "/" << cpu << "/" << clamp << "/" << mask << "/" << dest << "/"
                           << layout[0] << layout[1] << layout[2]);
              uint32_t words[4][34], want[4][34];
              std::mt19937 random(913);
              for (auto &reg : words) {
                std::fill(reg, reg + 34, 0xdeadbeef);
                for (int lane = 1; lane <= 32; ++lane)
                  reg[lane] = random();
              }
              std::memcpy(want, words, sizeof(want));
              for (int lane = 0; lane < 32; ++lane)
                if (mask >> lane & 1)
                  want[dest][lane + 1] =
                      reference(op, words[layout[0]][lane + 1], words[layout[1]][lane + 1],
                                words[layout[2]][lane + 1], clamp);
              uint32_t *a = words[layout[0]] + 1, *b = words[layout[1]] + 1,
                       *c = words[layout[2]] + 1, *d = words[dest] + 1;
              ASSERT_EQ(functions[op](cpu, mask, clamp ? GOC_ALU_CLAMP : 0, &d, &a, &b, &c),
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
        ASSERT_EQ(
            functions[test.op](cpu, UINT32_MAX, clamp ? GOC_ALU_CLAMP : 0, p + 3, p, p + 1, p + 2),
            GOC_SUCCESS);
        for (uint32_t word : words[3])
          EXPECT_EQ(word, clamp ? test.clamp : test.wrap);
      }
}

TEST(IntegerAdd, ValidationAndFpEnvironment) {
  std::fenv_t saved;
  std::fegetenv(&saved);
  for (int op = 0; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t input[32], output[32];
      std::fill(input, input + 32, 0x7f800001);
      std::fill(output, output + 32, 0xdeadbeef);
      auto a = input, d = output;
      for (uint64_t mask : {UINT64_C(0), UINT64_C(0xffffffff00000000), UINT64_MAX}) {
        for (int bit = 0; bit < 32; ++bit) {
          uint32_t mode = UINT32_C(1) << bit;
          if (op != 5 && mode == GOC_ALU_CLAMP)
            continue;
          EXPECT_EQ(functions[op](cpu, mask, mode, &d, &a, &a, &a), GOC_ERROR_INVALID_FLAGS);
        }
        EXPECT_EQ(functions[op](cpu | (UINT64_C(1) << 63), mask, 0, &d, &a, &a, &a),
                  GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(functions[op](cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, mask, 0,
                                &d, &a, &a, &a),
                  GOC_ERROR_UNSUPPORTED_SEMANTICS);
      }
      for (uint32_t word : output)
        EXPECT_EQ(word, 0xdeadbeef);
      for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        std::fesetround(rounding);
        std::feclearexcept(FE_ALL_EXCEPT);
        std::feraiseexcept(FE_DIVBYZERO);
        EXPECT_EQ(functions[op](cpu | GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX,
                                op == 5 ? 0 : GOC_ALU_CLAMP, &d, &a, &a, &a),
                  GOC_SUCCESS);
        EXPECT_EQ(std::fegetround(), rounding);
        EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
      }
    }
  std::fesetenv(&saved);
}

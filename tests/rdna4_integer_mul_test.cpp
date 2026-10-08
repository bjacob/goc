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

using Fn = decltype(&goc_rdna4_v_mul_lo_u32);
const Fn functions[] = {goc_rdna4_v_mul_lo_u32,     goc_rdna4_v_mul_hi_u32,
                        goc_rdna4_v_mul_hi_i32,     goc_rdna4_v_mul_i32_i24,
                        goc_rdna4_v_mul_hi_i32_i24, goc_rdna4_v_mul_u32_u24,
                        goc_rdna4_v_mul_hi_u32_u24};

bool can_clamp(int op) { return op == 3 || op == 5; }

uint32_t reference(int op, uint32_t a, uint32_t b, bool clamp) {
  const bool signed_op = op == 2 || op == 3 || op == 4;
  const bool high = op == 1 || op == 2 || op == 4 || op == 6;
  const int width = op < 3 ? 32 : 24;
  uint64_t modulus = UINT64_C(1) << width;
  a %= modulus;
  b %= modulus;
  bool negative_a = signed_op && a >= modulus / 2;
  bool negative_b = signed_op && b >= modulus / 2;
  uint64_t magnitude_a = negative_a ? modulus - a : a;
  uint64_t magnitude_b = negative_b ? modulus - b : b;
  // Multiply magnitudes; restore the sign after applying the range limit.
  uint64_t product = magnitude_a * magnitude_b;
  bool negative = negative_a != negative_b;
  if (clamp)
    product = std::min(product, signed_op ? (negative ? UINT64_C(0x80000000) : UINT64_C(0x7fffffff))
                                          : uint64_t(UINT32_MAX));
  if (negative)
    product = UINT64_C(0) - product;
  return uint32_t(high ? product >> 32 : product);
}

} // namespace

TEST(IntegerMul, BoundaryProductsAndRandomInputs) {
  const uint32_t boundary[] = {0,          1,          2,          46340,      46341,
                               65535,      65536,      0x007ffffe, 0x007fffff, 0x00800000,
                               0x00800001, 0x00ffffff, 0x01000000, 0x01000001, 0x7fffffff,
                               0x80000000, 0x80000001, 0xffffffff, 0xff7fffff, 0xff800000};
  for (int op = 0; op < 7; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int clamp = 0; clamp <= int(can_clamp(op)); ++clamp) {
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
          ASSERT_EQ(functions[op](cpu, UINT32_MAX, clamp ? GOC_ALU_CLAMP : 0, &pd, &pa, &pb),
                    GOC_SUCCESS);
          for (int lane = 0; lane < 32; ++lane)
            EXPECT_EQ(d[lane], reference(op, a[lane], b[lane], clamp));
        }
      }
}

TEST(IntegerMul, MasksAliasesAndSaturation) {
  const int layouts[][2] = {{0, 1}, {0, 0}, {1, 0}};
  for (int op = 0; op < 7; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int clamp = 0; clamp <= int(can_clamp(op)); ++clamp)
        for (uint64_t mask : rdna4_exec_masks())
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
                  want[dest][lane + 1] =
                      reference(op, words[layout[0]][lane + 1], words[layout[1]][lane + 1], clamp);
              uint32_t *a = words[layout[0]] + 1, *b = words[layout[1]] + 1, *d = words[dest] + 1;
              ASSERT_EQ(functions[op](cpu, mask, clamp ? GOC_ALU_CLAMP : 0, &d, &a, &b),
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
      ASSERT_EQ(functions[test.op](cpu, UINT32_MAX, test.mode, &pd, &pa, &pb), GOC_SUCCESS);
      for (uint32_t value : d)
        EXPECT_EQ(value, test.expected);
    }
}

TEST(IntegerMul, ValidationAndFpEnvironment) {
  std::fenv_t saved;
  std::fegetenv(&saved);
  for (int op = 0; op < 7; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t a[32], b[32], d[32];
      auto pa = a, pb = b, pd = d;
      std::fill(a, a + 32, 0x7f800001);
      std::fill(b, b + 32, 0xffffffff);
      std::fill(d, d + 32, 0xdeadbeef);
      for (uint64_t mask : {UINT64_C(0), UINT64_C(0xffffffff00000000), UINT64_MAX}) {
        for (int bit = 0; bit < 32; ++bit) {
          uint32_t mode = UINT32_C(1) << bit;
          if (mode == GOC_ALU_CLAMP && can_clamp(op))
            continue;
          EXPECT_EQ(functions[op](cpu, mask, mode, &pd, &pa, &pb), GOC_ERROR_INVALID_FLAGS);
        }
        EXPECT_EQ(functions[op](cpu | (UINT64_C(1) << 63), mask, 0, &pd, &pa, &pb),
                  GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(functions[op](cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, mask, 0,
                                &pd, &pa, &pb),
                  GOC_ERROR_UNSUPPORTED_SEMANTICS);
      }
      for (auto value : d)
        EXPECT_EQ(value, 0xdeadbeef);
      for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        std::fesetround(rounding);
        std::feclearexcept(FE_ALL_EXCEPT);
        std::feraiseexcept(FE_DIVBYZERO);
        EXPECT_EQ(functions[op](cpu | GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX,
                                can_clamp(op) ? GOC_ALU_CLAMP : 0, &pd, &pa, &pb),
                  GOC_SUCCESS);
        EXPECT_EQ(std::fegetround(), rounding);
        EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
      }
    }
  std::fesetenv(&saved);
}

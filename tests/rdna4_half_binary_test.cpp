// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_exec_masks.h"
#include "rdna4_half_reference.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <random>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_rdna4_v_add_f16);
const Fn functions[] = {goc_rdna4_v_add_f16,     goc_rdna4_v_sub_f16,     goc_rdna4_v_subrev_f16,
                        goc_rdna4_v_mul_f16,     goc_rdna4_v_min_num_f16, goc_rdna4_v_max_num_f16,
                        goc_rdna4_v_minimum_f16, goc_rdna4_v_maximum_f16};

const uint32_t known = GOC_ALU_NEG_A | GOC_ALU_NEG_B | GOC_ALU_ABS_A | GOC_ALU_ABS_B |
                       GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP | GOC_ALU_HIGH_A | GOC_ALU_HIGH_B |
                       GOC_ALU_HIGH_D;

uint16_t reference(int op, uint32_t a, uint32_t b, uint32_t mode, bool saturate) {
  double x = goc_test::half_value(uint16_t(a >> (mode & GOC_ALU_HIGH_A ? 16 : 0)));
  double y = goc_test::half_value(uint16_t(b >> (mode & GOC_ALU_HIGH_B ? 16 : 0)));
  if (mode & GOC_ALU_ABS_A)
    x = std::abs(x);
  if (mode & GOC_ALU_ABS_B)
    y = std::abs(y);
  if (mode & GOC_ALU_NEG_A)
    x = -x;
  if (mode & GOC_ALU_NEG_B)
    y = -y;
  double result = goc_test::half_binary(op, x, y);
  const double scales[] = {1, 2, 4, 0.5};
  result *= scales[(mode >> 6) & 3];
  if (mode & GOC_ALU_CLAMP)
    result = !(result > 0) ? 0 : std::min(result, 1.0);
  return goc_test::half_bits(result, saturate);
}

void check(uint32_t actual, uint32_t before, uint16_t want, uint32_t mode) {
  int shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  EXPECT_EQ((actual ^ before) & ~(UINT32_C(0xffff) << shift), 0u);
  uint16_t got = uint16_t(actual >> shift);
  if ((want & 0x7fff) > 0x7c00) {
    EXPECT_GT(got & 0x7fff, 0x7c00);
  } else {
    EXPECT_EQ(got, want);
  }
}

uint32_t modifiers(unsigned variant) {
  return (variant & 3) | ((variant & 12) << 1) | ((variant & 112) << 2) |
         (variant & 128 ? GOC_ALU_HIGH_A : 0) | (variant & 256 ? GOC_ALU_HIGH_B : 0) |
         (variant & 512 ? GOC_ALU_HIGH_D : 0);
}

} // namespace

TEST(HalfBinary, EveryEncodingAndRandomInputs) {
  for (int op = 0; op < 8; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool saturate : {false, true}) {
        std::mt19937 random(427);
        for (unsigned base = 0; base < 131072; base += 32) {
          SCOPED_TRACE(::testing::Message() << op << '/' << cpu << '/' << saturate << '/' << base);
          uint32_t words[3][32];
          uint32_t *p[] = {words[0], words[1], words[2]};
          for (int lane = 0; lane < 32; ++lane) {
            unsigned i = base + lane;
            words[0][lane] = i < 65536 ? i : random();
            words[1][lane] = random();
            words[2][lane] = 0xdeadbeef;
          }
          ASSERT_EQ(
              functions[op](cpu | (saturate ? GOC_FP16_OVFL : 0), UINT32_MAX, 0, p + 2, p, p + 1),
              GOC_SUCCESS);
          for (int lane = 0; lane < 32; ++lane)
            check(words[2][lane], 0xdeadbeef,
                  reference(op, words[0][lane], words[1][lane], 0, saturate), 0);
        }
      }
}

TEST(HalfBinary, AllModifiersAndHalfSelectors) {
  const uint16_t values[] = {0,      0x8000, 1,      0x8001, 0x3ff,  0x400,  0x3bff, 0x3c00,
                             0xbc00, 0x4001, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7c01, 0xfe00};
  for (int op = 0; op < 8; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < 1024; ++variant)
        for (bool saturate : {false, true}) {
          auto mode = modifiers(variant);
          SCOPED_TRACE(::testing::Message() << op << '/' << cpu << '/' << mode);
          uint32_t words[3][32], before[3][32];
          for (int reg = 0; reg < 3; ++reg)
            for (int lane = 0; lane < 32; ++lane)
              words[reg][lane] = values[(lane + reg * 3) % 16] |
                                 (uint32_t(values[(lane / 2 + reg * 5) % 16]) << 16);
          std::memcpy(before, words, sizeof(words));
          int dest = variant % 3;
          uint64_t mask = UINT32_MAX;
          uint32_t *p[] = {words[0], words[1], words[2]};
          ASSERT_EQ(
              functions[op](cpu | (saturate ? GOC_FP16_OVFL : 0), mask, mode, p + dest, p, p + 1),
              GOC_SUCCESS);
          for (int reg = 0; reg < 3; ++reg)
            for (int lane = 0; lane < 32; ++lane) {
              if (reg == dest && ((mask >> lane) & 1)) {
                check(words[reg][lane], before[reg][lane],
                      reference(op, before[0][lane], before[1][lane], mode, saturate), mode);
              } else {
                EXPECT_EQ(words[reg][lane], before[reg][lane]);
              }
            }
        }
}

TEST(HalfBinary, MasksAliasesAndUntouchedHalves) {
  const uint32_t arithmetic_modes[] = {0, GOC_ALU_NEG_A | GOC_ALU_ABS_A | GOC_ALU_NEG_B,
                                       GOC_ALU_OMOD_HALF, GOC_ALU_OMOD_4 | GOC_ALU_CLAMP};
  for (int op = 0; op < 8; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned selectors = 0; selectors < 8; ++selectors)
        for (uint32_t arithmetic : arithmetic_modes)
          for (uint64_t mask : rdna4_exec_masks())
            for (int b = 0; b < 2; ++b)
              for (int dest = 0; dest < 3; ++dest) {
                uint32_t mode = modifiers(selectors << 7) | arithmetic;
                SCOPED_TRACE(::testing::Message() << op << '/' << cpu << '/' << mode << '/' << mask
                                                  << '/' << b << '/' << dest);
                uint32_t words[3][34], before[3][34];
                std::mt19937 random(12);
                for (auto &reg : words)
                  for (auto &word : reg)
                    word = random();
                std::memcpy(before, words, sizeof(words));
                uint32_t *p[] = {words[0] + 1, words[1] + 1, words[2] + 1};
                ASSERT_EQ(functions[op](cpu, mask, mode, p + dest, p, p + b), GOC_SUCCESS);
                for (int reg = 0; reg < 3; ++reg)
                  for (int lane = 0; lane < 34; ++lane) {
                    if (reg == dest && lane >= 1 && lane <= 32 && ((mask >> (lane - 1)) & 1)) {
                      check(words[reg][lane], before[reg][lane],
                            reference(op, before[0][lane], before[b][lane], mode, false), mode);
                    } else {
                      EXPECT_EQ(words[reg][lane], before[reg][lane]);
                    }
                  }
              }
}

TEST(HalfBinary, LiteralRoundingOverflowAndNanRules) {
  struct Case {
    int op;
    uint16_t a, b, result;
    uint32_t mode;
    bool saturate;
  };

  const Case cases[] = {{0, 0x3c00, 0x1000, 0x3c00, 0, false}, // Even tie rounds down.
                        {0, 0x3c01, 0x1000, 0x3c02, 0, false}, // Odd tie rounds up.
                        {1, 0x0400, 0x03ff, 1, 0, false},
                        {2, 0x3c00, 0x4000, 0x3c00, 0, false},
                        {3, 1, 0x3800, 0, 0, false},
                        {3, 3, 0x3800, 2, 0, false},
                        {3, 0x8001, 0x3800, 0x8000, 0, false},
                        {0, 0x7bff, 0x7bff, 0x7c00, 0, false},
                        {0, 0x7bff, 0x7bff, 0x7bff, 0, true},
                        {0, 0xfbff, 0xfbff, 0xfbff, 0, true},
                        {0, 0x7bff, 0x7bff, 0x7bff, GOC_ALU_OMOD_HALF, false},
                        {3, 0x7c00, 0x3c00, 0x7c00, 0, true},
                        {4, 0x8000, 0, 0x8000, 0, false},
                        {5, 0x8000, 0, 0, 0, false},
                        {6, 0x8000, 0, 0x8000, 0, false},
                        {7, 0x8000, 0, 0, 0, false},
                        {4, 0x7c01, 0x3c00, 0x3c00, 0, false},
                        {5, 0x3c00, 0xfc01, 0x3c00, 0, false},
                        {6, 0x7c01, 0x3c00, 0x7e00, 0, false},
                        {7, 0x3c00, 0x7e00, 0, GOC_ALU_CLAMP, false},
                        {3, 0, 0x7c00, 0, GOC_ALU_CLAMP, false}};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (const auto &test : cases) {
      uint32_t words[3][32];
      uint32_t *p[] = {words[0], words[1], words[2]};
      std::fill(words[0], words[0] + 32, test.a);
      std::fill(words[1], words[1] + 32, test.b);
      std::fill(words[2], words[2] + 32, 0xdeadbeef);
      ASSERT_EQ(functions[test.op](cpu | (test.saturate ? GOC_FP16_OVFL : 0), UINT32_MAX, test.mode,
                                   p + 2, p, p + 1),
                GOC_SUCCESS);
      for (auto word : words[2])
        check(word, 0xdeadbeef, test.result, test.mode);
    }
}

TEST(HalfBinary, ValidationAndSemantics) {
  for (auto fn : functions) {
    uint32_t data[32];
    std::fill(data, data + 32, 0xdeadbeef);
    auto p = data;
    for (uint64_t mask : {UINT64_C(0), UINT64_C(0xffffffff00000000), UINT64_MAX}) {
      for (int bit = 0; bit < 32; ++bit)
        if (!(known & (UINT32_C(1) << bit))) {
          EXPECT_EQ(fn(0, mask, UINT32_C(1) << bit, &p, &p, &p), GOC_ERROR_INVALID_FLAGS);
        }
      EXPECT_EQ(fn(UINT64_C(1) << 63, mask, 0, &p, &p, &p), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, mask, 0, &p, &p, &p),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
    }
    for (auto word : data)
      EXPECT_EQ(word, 0xdeadbeef);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, 0, &p, &p, &p), GOC_SUCCESS);
  }
}

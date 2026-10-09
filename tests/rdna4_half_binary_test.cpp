// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_exec_masks.h"
#include "rdna4_half_binary_reference.h"
#include "rdna4_half_reference.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <random>
#include <stdint.h>

namespace {

void check(uint32_t actual, uint32_t before, uint16_t want, uint32_t mode) {
  int shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  EXPECT_EQ((actual ^ before) & ~(0xffffU << shift), 0u);
  uint16_t got = uint16_t(actual >> shift);
  if ((want & 0x7fff) > 0x7c00) {
    EXPECT_GT(got & 0x7fff, 0x7c00);
  } else {
    EXPECT_EQ(got, want);
  }
}

} // namespace

TEST(HalfBinary, CanonicalNanPreservesUnselectedHalf) {
  for (uint32_t bits = 0; bits < 65536; ++bits) {
    const bool nan = (bits & 0x7c00) == 0x7c00 && (bits & 0x3ff) != 0;
    const uint32_t expected = nan ? 0x7e00 : bits;
    // A signaling NaN in the other half must remain untouched.
    EXPECT_EQ(goc_test::canonical_half_nan(0xfc010000U | bits, 0), 0xfc010000U | expected);
    EXPECT_EQ(goc_test::canonical_half_nan((bits << 16) | 0x7c01, GOC_ALU_HIGH_D),
              (expected << 16) | 0x7c01);
  }
}

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
          ASSERT_EQ(goc_test::half_binary_functions[op](cpu | (saturate ? GOC_FP16_OVFL : 0),
                                                        UINT32_MAX, 0, p + 2, p, p + 1),
                    GOC_SUCCESS);
          for (int lane = 0; lane < 32; ++lane)
            check(words[2][lane], 0xdeadbeef,
                  goc_test::half_binary_reference(op, words[0][lane], words[1][lane], 0, saturate),
                  0);
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
          auto mode = goc_test::half_binary_modifiers(variant);
          SCOPED_TRACE(::testing::Message() << op << '/' << cpu << '/' << mode);
          uint32_t words[3][32], before[3][32];
          for (int reg = 0; reg < 3; ++reg)
            for (int lane = 0; lane < 32; ++lane)
              words[reg][lane] = values[(lane + reg * 3) % 16] |
                                 (uint32_t(values[(lane / 2 + reg * 5) % 16]) << 16);
          std::memcpy(before, words, sizeof(words));
          int dest = variant % 3;
          uint32_t exec_mask = UINT32_MAX;
          uint32_t *p[] = {words[0], words[1], words[2]};
          ASSERT_EQ(goc_test::half_binary_functions[op](cpu | (saturate ? GOC_FP16_OVFL : 0),
                                                        exec_mask, mode, p + dest, p, p + 1),
                    GOC_SUCCESS);
          for (int reg = 0; reg < 3; ++reg)
            for (int lane = 0; lane < 32; ++lane) {
              if (reg == dest && ((exec_mask >> lane) & 1)) {
                check(words[reg][lane], before[reg][lane],
                      goc_test::half_binary_reference(op, before[0][lane], before[1][lane], mode,
                                                      saturate),
                      mode);
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
          for (uint32_t exec_mask : rdna4_exec_masks())
            for (int b = 0; b < 2; ++b)
              for (int dest = 0; dest < 3; ++dest) {
                uint32_t mode = goc_test::half_binary_modifiers(selectors << 7) | arithmetic;
                SCOPED_TRACE(::testing::Message() << op << '/' << cpu << '/' << mode << '/'
                                                  << exec_mask << '/' << b << '/' << dest);
                uint32_t words[3][34], before[3][34];
                std::mt19937 random(12);
                for (auto &reg : words)
                  for (auto &word : reg)
                    word = random();
                std::memcpy(before, words, sizeof(words));
                uint32_t *p[] = {words[0] + 1, words[1] + 1, words[2] + 1};
                ASSERT_EQ(
                    goc_test::half_binary_functions[op](cpu, exec_mask, mode, p + dest, p, p + b),
                    GOC_SUCCESS);
                for (int reg = 0; reg < 3; ++reg)
                  for (int lane = 0; lane < 34; ++lane) {
                    if (reg == dest && lane >= 1 && lane <= 32 && ((exec_mask >> (lane - 1)) & 1)) {
                      check(words[reg][lane], before[reg][lane],
                            goc_test::half_binary_reference(op, before[0][lane], before[b][lane],
                                                            mode, false),
                            mode);
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
      ASSERT_EQ(goc_test::half_binary_functions[test.op](cpu | (test.saturate ? GOC_FP16_OVFL : 0),
                                                         UINT32_MAX, test.mode, p + 2, p, p + 1),
                GOC_SUCCESS);
      for (auto word : words[2])
        check(word, 0xdeadbeef, test.result, test.mode);
    }
}

TEST(HalfBinary, ValidationAndSemantics) {
  for (auto fn : goc_test::half_binary_functions) {
    uint32_t data[32];
    std::fill(data, data + 32, 0xdeadbeef);
    auto p = data;
    for (uint32_t exec_mask : {0U, UINT32_MAX}) {
      for (int bit = 0; bit < 32; ++bit)
        if (!(goc_test::half_binary_known & (1U << bit))) {
          EXPECT_EQ(fn(0, exec_mask, 1U << bit, &p, &p, &p), GOC_ERROR_INVALID_FLAGS);
        }
      EXPECT_EQ(fn(1ULL << 63, exec_mask, 0, &p, &p, &p), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, exec_mask, 0, &p, &p, &p),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
    }
    for (auto word : data)
      EXPECT_EQ(word, 0xdeadbeef);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, 0, &p, &p, &p), GOC_SUCCESS);
  }
}

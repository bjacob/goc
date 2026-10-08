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

using Fn = decltype(&goc_rdna4_v_min3_num_f16);
const Fn functions[] = {
    goc_rdna4_v_min3_num_f16,       goc_rdna4_v_max3_num_f16,       goc_rdna4_v_minmax_num_f16,
    goc_rdna4_v_maxmin_num_f16,     goc_rdna4_v_minimum3_f16,       goc_rdna4_v_maximum3_f16,
    goc_rdna4_v_minimummaximum_f16, goc_rdna4_v_maximumminimum_f16, goc_rdna4_v_med3_num_f16};
const uint16_t values[] = {0,      0x8000, 1,      0x8001, 0x3ff,  0x400,  0x3800, 0xb800,
                           0x3bff, 0x3c00, 0x3e00, 0xbe00, 0x4100, 0xc100, 0x4200, 0x4400,
                           0x4c00, 0xcc00, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7c01, 0xfe00};

bool nan(uint16_t bits) { return (bits & 0x7fff) > 0x7c00; }

uint16_t ordered(uint16_t bits) { return bits & 0x8000 ? uint16_t(~bits) : bits ^ 0x8000; }

uint16_t select(uint16_t a, uint16_t b, bool maximum, bool propagate) {
  if (nan(a) || nan(b)) {
    if (propagate || (nan(a) && nan(b)))
      return 0x7e00;
    return nan(a) ? b : a;
  }
  return (maximum ? ordered(a) > ordered(b) : ordered(a) < ordered(b)) ? a : b;
}

uint16_t reference(int op, uint32_t a, uint32_t b, uint32_t c, uint32_t mode, bool saturate) {
  const uint32_t words[] = {a, b, c};
  uint16_t input[3];
  for (int i = 0; i < 3; ++i) {
    input[i] = uint16_t(words[i] >> (mode & (GOC_ALU_HIGH_A << i) ? 16 : 0));
    if (mode & (GOC_ALU_ABS_A << i))
      input[i] &= 0x7fff;
    if (mode & (GOC_ALU_NEG_A << i))
      input[i] ^= 0x8000;
  }
  bool first_maximum = op % 4 == 1 || op % 4 == 3;
  bool second_maximum = op % 4 == 1 || op % 4 == 2;
  uint16_t result =
      select(select(input[0], input[1], first_maximum, op >= 4), input[2], second_maximum, op >= 4);
  if (op == 8) {
    if (nan(input[0]) || nan(input[1]) || nan(input[2])) {
      result = select(select(input[0], input[1], false, false), input[2], false, false);
    } else {
      uint16_t sorted[] = {input[0], input[1], input[2]};
      std::sort(sorted, sorted + 3, [](uint16_t a, uint16_t b) { return ordered(a) < ordered(b); });
      result = sorted[1];
      // Numeric equality identifies the first maximum even across signed zeros.
      if ((result & 0x7fff) == 0) {
        double maximum = goc_test::half_value(sorted[2]);
        int drop = goc_test::half_value(input[0]) == maximum   ? 0
                   : goc_test::half_value(input[1]) == maximum ? 1
                                                               : 2;
        result = select(input[(drop + 1) % 3], input[(drop + 2) % 3], true, false);
      }
    }
  }
  const double scales[] = {1, 2, 4, 0.5};
  double value = goc_test::half_value(result) * scales[(mode >> 6) & 3];
  if (mode & GOC_ALU_CLAMP)
    value = !(value > 0) ? 0 : std::min(value, 1.0);
  return goc_test::half_bits(value, saturate);
}

void check(uint32_t actual, uint32_t before, uint16_t want, uint32_t mode) {
  int shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  EXPECT_EQ((actual ^ before) & ~(UINT32_C(0xffff) << shift), 0u);
  uint16_t got = uint16_t(actual >> shift);
  if (nan(want)) {
    EXPECT_TRUE(nan(got));
  } else {
    EXPECT_EQ(got, want);
  }
}

} // namespace

TEST(HalfMinmax3, BoundaryCartesianProductsAndEveryEncoding) {
  for (int op = 0; op < 9; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool high : {false, true}) {
        std::mt19937 random(472);
        uint32_t mode =
            high ? GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_C | GOC_ALU_HIGH_D : 0;
        for (unsigned base = 0; base < 13824 + 65536; base += 32) {
          SCOPED_TRACE(::testing::Message() << op << '/' << cpu << '/' << high << '/' << base);
          uint32_t words[4][32];
          uint32_t *p[] = {words[0], words[1], words[2], words[3]};
          for (int lane = 0; lane < 32; ++lane) {
            unsigned i = base + lane;
            for (int reg = 0; reg < 3; ++reg) {
              uint16_t code = base < 13824 ? values[i % 24]
                              : reg == 0   ? uint16_t(i - 13824)
                                           : uint16_t(random());
              if (base < 13824)
                i /= 24;
              words[reg][lane] = high ? (uint32_t(code) << 16) | 0xbeef : 0xdead0000 | code;
            }
            words[3][lane] = 0xdeadbeef;
          }
          ASSERT_EQ(functions[op](cpu, UINT32_MAX, mode, p + 3, p, p + 1, p + 2), GOC_SUCCESS);
          for (int lane = 0; lane < 32; ++lane)
            check(words[3][lane], 0xdeadbeef,
                  reference(op, words[0][lane], words[1][lane], words[2][lane], mode, false), mode);
        }
      }
}

TEST(HalfMinmax3, AllModifiersAndHalfSelectors) {
  for (int op = 0; op < 9; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint32_t mode = 0; mode < 8192; ++mode)
        for (bool saturate : {false, true}) {
          SCOPED_TRACE(::testing::Message() << op << '/' << cpu << '/' << mode << '/' << saturate);
          uint32_t words[4][32], before[4][32];
          for (int reg = 0; reg < 4; ++reg)
            for (int lane = 0; lane < 32; ++lane)
              words[reg][lane] = values[(lane + reg * 5) % 24] |
                                 (uint32_t(values[(lane * 7 + reg * 3) % 24]) << 16);
          std::memcpy(before, words, sizeof(words));
          int dest = mode % 4;
          uint32_t *p[] = {words[0], words[1], words[2], words[3]};
          ASSERT_EQ(functions[op](cpu | (saturate ? GOC_FP16_OVFL : 0), UINT32_MAX, mode, p + dest,
                                  p, p + 1, p + 2),
                    GOC_SUCCESS);
          for (int reg = 0; reg < 4; ++reg)
            for (int lane = 0; lane < 32; ++lane) {
              if (reg == dest) {
                check(words[reg][lane], before[reg][lane],
                      reference(op, before[0][lane], before[1][lane], before[2][lane], mode,
                                saturate),
                      mode);
              } else {
                EXPECT_EQ(words[reg][lane], before[reg][lane]);
              }
            }
        }
}

TEST(HalfMinmax3, MasksAndAllWholeRegisterAliases) {
  const int layouts[][3] = {{0, 1, 2}, {0, 0, 2}, {0, 1, 0}, {0, 1, 1}, {0, 0, 0}};
  const uint32_t modes[] = {
      0, UINT32_C(63) | GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_C | GOC_ALU_HIGH_D,
      GOC_ALU_HIGH_A | GOC_ALU_HIGH_C | GOC_ALU_HIGH_D | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP,
      GOC_ALU_HIGH_B | GOC_ALU_OMOD_4};
  for (int op = 0; op < 9; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (auto mode : modes)
        for (auto mask : rdna4_exec_masks())
          for (const auto &layout : layouts)
            for (int dest = 0; dest < 4; ++dest) {
              SCOPED_TRACE(::testing::Message()
                           << op << '/' << cpu << '/' << mode << '/' << mask << '/' << dest << '/'
                           << layout[0] << layout[1] << layout[2]);
              uint32_t words[4][34], before[4][34];
              for (int reg = 0; reg < 4; ++reg) {
                std::fill(words[reg], words[reg] + 34, 0xdeadbeef);
                for (int lane = 1; lane <= 32; ++lane)
                  words[reg][lane] = values[(lane + reg * 5) % 24] |
                                     (uint32_t(values[(lane * 7 + reg * 3) % 24]) << 16);
              }
              std::memcpy(before, words, sizeof(words));
              uint32_t *p[] = {words[0] + 1, words[1] + 1, words[2] + 1, words[3] + 1};
              ASSERT_EQ(functions[op](cpu, mask, mode, p + dest, p + layout[0], p + layout[1],
                                      p + layout[2]),
                        GOC_SUCCESS);
              for (int reg = 0; reg < 4; ++reg)
                for (int lane = 0; lane < 34; ++lane) {
                  if (reg == dest && lane >= 1 && lane <= 32 && ((mask >> (lane - 1)) & 1)) {
                    check(words[reg][lane], before[reg][lane],
                          reference(op, before[layout[0]][lane], before[layout[1]][lane],
                                    before[layout[2]][lane], mode, false),
                          mode);
                  } else {
                    EXPECT_EQ(words[reg][lane], before[reg][lane]);
                  }
                }
            }
}

TEST(HalfMinmax3, LiteralOrderNanAndSignedZeroRules) {
  struct Case {
    uint16_t a, b, c, expected[9];
  };

  const Case cases[] = {
      {0x3c00,
       0x4200,
       0x4000,
       {0x3c00, 0x4200, 0x4000, 0x4000, 0x3c00, 0x4200, 0x4000, 0x4000, 0x4000}},
      {0x4200,
       0x3c00,
       0x4000,
       {0x3c00, 0x4200, 0x4000, 0x4000, 0x3c00, 0x4200, 0x4000, 0x4000, 0x4000}},
      {0x7c01,
       0x3c00,
       0x4000,
       {0x3c00, 0x4000, 0x4000, 0x3c00, 0x7e00, 0x7e00, 0x7e00, 0x7e00, 0x3c00}},
      {0x3c00,
       0xfe00,
       0x4000,
       {0x3c00, 0x4000, 0x4000, 0x3c00, 0x7e00, 0x7e00, 0x7e00, 0x7e00, 0x3c00}},
      {0x3c00,
       0x4000,
       0xfe01,
       {0x3c00, 0x4000, 0x3c00, 0x4000, 0x7e00, 0x7e00, 0x7e00, 0x7e00, 0x3c00}},
      {0x8000, 0, 0x8000, {0x8000, 0, 0x8000, 0x8000, 0x8000, 0, 0x8000, 0x8000, 0}},
      {0, 0x8000, 0x8000, {0x8000, 0, 0x8000, 0x8000, 0x8000, 0, 0x8000, 0x8000, 0x8000}}};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (const auto &test : cases)
      for (int op = 0; op < 9; ++op) {
        uint32_t words[4][32];
        uint32_t *p[] = {words[0], words[1], words[2], words[3]};
        std::fill(words[0], words[0] + 32, test.a);
        std::fill(words[1], words[1] + 32, test.b);
        std::fill(words[2], words[2] + 32, test.c);
        std::fill(words[3], words[3] + 32, 0xdeadbeef);
        ASSERT_EQ(functions[op](cpu, UINT32_MAX, 0, p + 3, p, p + 1, p + 2), GOC_SUCCESS);
        for (auto word : words[3])
          check(word, 0xdeadbeef, test.expected[op], 0);
      }
}

TEST(HalfMinmax3, ValidationAndSemantics) {
  for (auto fn : functions) {
    uint32_t data[32];
    std::fill(data, data + 32, 0xdeadbeef);
    auto p = data;
    for (uint64_t mask : {UINT64_C(0), UINT64_C(0xffffffff00000000), UINT64_MAX}) {
      for (int bit = 13; bit < 32; ++bit)
        EXPECT_EQ(fn(0, mask, UINT32_C(1) << bit, &p, &p, &p, &p), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(UINT64_C(1) << 63, mask, 0, &p, &p, &p, &p), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, mask, 0, &p, &p, &p, &p),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
    }
    for (auto word : data)
      EXPECT_EQ(word, 0xdeadbeef);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, 0, &p, &p, &p, &p), GOC_SUCCESS);
  }
}

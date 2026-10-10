// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "dpp_reference.h"
#include "exec_masks.h"
#include "goc/goc.h"
#include "half_minmax_reference.h"
#include "half_reference.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <random>
#include <stdint.h>

namespace {

const decltype(&goc_v_min3_num_f16_wave64) wave64[] = {
    goc_v_min3_num_f16_wave64,       goc_v_max3_num_f16_wave64,
    goc_v_minmax_num_f16_wave64,     goc_v_maxmin_num_f16_wave64,
    goc_v_minimum3_f16_wave64,       goc_v_maximum3_f16_wave64,
    goc_v_minimummaximum_f16_wave64, goc_v_maximumminimum_f16_wave64,
    goc_v_med3_num_f16_wave64,       goc_v_min3_f16_wave64,
    goc_v_max3_f16_wave64,           goc_v_minmax_f16_wave64,
    goc_v_maxmin_f16_wave64,         goc_v_med3_f16_wave64};

const uint16_t values[] = {0,      0x8000, 1,      0x8001, 0x3ff,  0x400,  0x3800, 0xb800,
                           0x3bff, 0x3c00, 0x3e00, 0xbe00, 0x4100, 0xc100, 0x4200, 0x4400,
                           0x4c00, 0xcc00, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7c01, 0xfe00};

void check(uint32_t actual, uint32_t before, uint16_t want, uint32_t mode) {
  int shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  EXPECT_EQ((actual ^ before) & ~(0xffffU << shift), 0u);
  uint16_t got = uint16_t(actual >> shift);
  if (goc_test::half_minmax_nan(want)) {
    EXPECT_TRUE(goc_test::half_minmax_nan(got));
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
          ASSERT_EQ(goc_test::half_minmax_functions[op](cpu, UINT32_MAX, mode, p + 3, p, p + 1,
                                                        p + 2, nullptr),
                    GOC_SUCCESS);
          for (int lane = 0; lane < 32; ++lane)
            check(words[3][lane], 0xdeadbeef,
                  goc_test::half_minmax_reference(op, words[0][lane], words[1][lane],
                                                  words[2][lane], mode, false),
                  mode);
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
          ASSERT_EQ(goc_test::half_minmax_functions[op](cpu | (saturate ? GOC_FP16_OVFL : 0),
                                                        UINT32_MAX, mode, p + dest, p, p + 1, p + 2,
                                                        nullptr),
                    GOC_SUCCESS);
          for (int reg = 0; reg < 4; ++reg)
            for (int lane = 0; lane < 32; ++lane) {
              if (reg == dest) {
                check(words[reg][lane], before[reg][lane],
                      goc_test::half_minmax_reference(op, before[0][lane], before[1][lane],
                                                      before[2][lane], mode, saturate),
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
      0, 63U | GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_C | GOC_ALU_HIGH_D,
      GOC_ALU_HIGH_A | GOC_ALU_HIGH_C | GOC_ALU_HIGH_D | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP,
      GOC_ALU_HIGH_B | GOC_ALU_OMOD_4};
  for (int op = 0; op < 9; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (auto mode : modes)
        for (auto exec_mask : exec_masks())
          for (const auto &layout : layouts)
            for (int dest = 0; dest < 4; ++dest) {
              SCOPED_TRACE(::testing::Message()
                           << op << '/' << cpu << '/' << mode << '/' << exec_mask << '/' << dest
                           << '/' << layout[0] << layout[1] << layout[2]);
              uint32_t words[4][34], before[4][34];
              for (int reg = 0; reg < 4; ++reg) {
                std::fill(words[reg], words[reg] + 34, 0xdeadbeef);
                for (int lane = 1; lane <= 32; ++lane)
                  words[reg][lane] = values[(lane + reg * 5) % 24] |
                                     (uint32_t(values[(lane * 7 + reg * 3) % 24]) << 16);
              }
              std::memcpy(before, words, sizeof(words));
              uint32_t *p[] = {words[0] + 1, words[1] + 1, words[2] + 1, words[3] + 1};
              ASSERT_EQ(goc_test::half_minmax_functions[op](cpu, exec_mask, mode, p + dest,
                                                            p + layout[0], p + layout[1],
                                                            p + layout[2], nullptr),
                        GOC_SUCCESS);
              for (int reg = 0; reg < 4; ++reg)
                for (int lane = 0; lane < 34; ++lane) {
                  if (reg == dest && lane >= 1 && lane <= 32 && ((exec_mask >> (lane - 1)) & 1)) {
                    check(words[reg][lane], before[reg][lane],
                          goc_test::half_minmax_reference(op, before[layout[0]][lane],
                                                          before[layout[1]][lane],
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
      {0x8000, 0, 0x8000, {0x8000, 0, 0x8000, 0x8000, 0x8000, 0, 0x8000, 0x8000, 0x8000}},
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
        ASSERT_EQ(goc_test::half_minmax_functions[op](cpu, UINT32_MAX, 0, p + 3, p, p + 1, p + 2,
                                                      nullptr),
                  GOC_SUCCESS);
        for (auto word : words[3])
          check(word, 0xdeadbeef, test.expected[op], 0);
      }
}

TEST(HalfMinmax3, ValidationAndSemantics) {
  for (auto fn : goc_test::half_minmax_functions) {
    uint32_t data[32];
    std::fill(data, data + 32, 0xdeadbeef);
    auto p = data;
    for (uint32_t exec_mask : {0U, UINT32_MAX}) {
      for (int bit = 13; bit < 32; ++bit)
        EXPECT_EQ(fn(0, exec_mask, 1U << bit, &p, &p, &p, &p, nullptr), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(1ULL << 63, exec_mask, 0, &p, &p, &p, &p, nullptr), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, exec_mask, 0, &p, &p, &p,
                   &p, nullptr),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
    }
    for (auto word : data)
      EXPECT_EQ(word, 0xdeadbeef);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, 0, &p, &p, &p, &p, nullptr),
              GOC_SUCCESS);
  }
}

TEST(HalfMinmax3, HardwareMedianSignedZeroOrdering) {
  const uint32_t values[] = {0, 0x8000, 0x3c00, 0xbc00, 1, 0x8001, 0x7c00, 0xfc00};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint64_t hash = goc_test::capture_hash_seed;
    for (unsigned base = 0; base < 512; base += 32) {
      uint32_t words[4][32];
      for (unsigned lane = 0; lane < 32; ++lane) {
        unsigned index = base + lane;
        words[0][lane] = values[index % 8];
        words[1][lane] = values[(index / 8) % 8];
        words[2][lane] = values[(index / 64) % 8];
        words[3][lane] = 0xdead0000u + lane;
      }
      uint32_t *p[] = {words[0], words[1], words[2], words[3]};
      ASSERT_EQ(goc_v_med3_num_f16(cpu, UINT32_MAX, 0, p + 3, p, p + 1, p + 2, nullptr),
                GOC_SUCCESS);
      for (uint32_t word : words[3])
        hash = goc_test::capture_hash_word(hash, word);
    }
    EXPECT_EQ(hash, 0x6621dd1b4ac695edULL);
  }
}

TEST(HalfMinmax3, HardwareMedianNanRules) {
  const uint32_t values[] = {0, 0x8000, 0x7c01, 0x7e00, 0xfc01, 0xfe00, 0x3c00, 0xbc00};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint64_t hash = goc_test::capture_hash_seed;
    for (unsigned base = 0; base < 512; base += 32) {
      uint32_t words[4][32];
      for (unsigned lane = 0; lane < 32; ++lane) {
        unsigned index = base + lane;
        words[0][lane] = values[index % 8];
        words[1][lane] = values[(index / 8) % 8];
        words[2][lane] = values[(index / 64) % 8];
        words[3][lane] = 0xdead0000u + lane;
      }
      uint32_t *p[] = {words[0], words[1], words[2], words[3]};
      ASSERT_EQ(goc_v_med3_num_f16(cpu, UINT32_MAX, 0, p + 3, p, p + 1, p + 2, nullptr),
                GOC_SUCCESS);
      for (uint32_t word : words[3]) {
        if ((word & 0x7fff) > 0x7c00)
          word = (word & 0xffff0000u) | 0x7e00;
        hash = goc_test::capture_hash_word(hash, word);
      }
    }
    EXPECT_EQ(hash, 0x513891779d603325ULL);
  }
}

TEST(HalfMinmax3, Wave64ModifiersDppSelectorsAndAliases) {

  const unsigned reference_ops[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 8};
  for (unsigned op = 0; op < 14; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint32_t mode = 0; mode < 8192; ++mode) {
        unsigned route_index = mode % 8;
        uint64_t descriptor = route_index ? goc_test::dpp_modes[route_index - 1] : 0;
        if (descriptor & GOC_DPP16)
          descriptor =
              (descriptor & ~GOC_DPP_ROW_MASK) | (uint64_t((mode >> 3) & 15) << GOC_DPP_ROW_SHIFT);
        uint32_t words[4][66], before[4][66];
        for (unsigned reg = 0; reg < 4; ++reg)
          for (unsigned lane = 0; lane < 66; ++lane)
            words[reg][lane] =
                values[(lane + reg * 5) % 24] | (uint32_t(values[(lane * 7 + reg * 3) % 24]) << 16);
        std::memcpy(before, words, sizeof words);
        unsigned dest = mode % 4;
        uint32_t *d = words[dest] + 1;
        const uint32_t *a = words[0] + 1, *b = words[1] + 1, *c = words[2] + 1;
        uint64_t exec_mask = 0x87654321fedcba98ULL;
        bool saturate = mode & 1;
        uint32_t exceptions = 0x80000055;
        ASSERT_EQ(wave64[op](cpu | (saturate ? GOC_FP16_OVFL : 0), exec_mask, descriptor | mode, &d,
                             &a, &b, &c, &exceptions),
                  GOC_SUCCESS);
        EXPECT_EQ(exceptions, 0x80000055U);
        for (unsigned lane = 0; lane < 64; ++lane) {
          int source = int(lane);
          bool enabled = descriptor ? goc_test::dpp_source(descriptor, exec_mask, lane, source)
                                    : ((exec_mask >> lane) & 1);
          if (enabled) {
            check(d[lane], before[dest][lane + 1],
                  goc_test::half_minmax_reference(
                      reference_ops[op], source < 0 ? 0 : before[0][source + 1],
                      before[1][lane + 1], before[2][lane + 1], mode, saturate, op >= 9),
                  mode);
          } else {
            EXPECT_EQ(d[lane], before[dest][lane + 1]);
          }
        }
        for (unsigned reg = 0; reg < 4; ++reg) {
          EXPECT_EQ(words[reg][0], before[reg][0]);
          EXPECT_EQ(words[reg][65], before[reg][65]);
        }
      }
}

TEST(HalfMinmax3, Wave64ValidationAndOptionalState) {
  for (auto fn : wave64) {
    EXPECT_EQ(fn(0, 0, 0, nullptr, nullptr, nullptr, nullptr, nullptr), GOC_SUCCESS);
    EXPECT_EQ(fn(0, UINT64_MAX, 1ULL << 36, nullptr, nullptr, nullptr, nullptr, nullptr),
              GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, UINT64_MAX, 0, nullptr,
                 nullptr, nullptr, nullptr, nullptr),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    uint32_t exceptions = 0x80000055;
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL, UINT64_MAX, 0, nullptr, nullptr, nullptr, nullptr,
                 &exceptions),
              GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
    EXPECT_EQ(exceptions, 0x80000055U);
  }
}

TEST(HalfMinmax3, ArchitectureGuestSelectionPayloadsAndZeroTies) {
  const goc_test::HalfMinmaxFn legacy[] = {goc_v_min3_f16, goc_v_max3_f16, goc_v_minmax_f16,
                                           goc_v_maxmin_f16, goc_v_med3_f16};

  struct Case {
    uint16_t a, b, c;
    bool ieee;
    uint16_t expected[5];
  };

  // RDNA3 nested binary selection, including first-maximum removal for MED3.
  const Case cases[] = {
      {0x7c15, 0x7e26, 0xfe37, false, {0x7c15, 0x7c15, 0x7c15, 0x7c15, 0x7c15}},
      {0x7c15, 0x7e26, 0xfe37, true, {0x7e15, 0x7e15, 0x7e15, 0x7e15, 0x7e15}},
      {0x3c00, 0x4000, 0xfc15, true, {0xfe15, 0xfe15, 0xfe15, 0xfe15, 0xfe15}},
      {0x3c00, 0x4000, 0xfc15, false, {0x3c00, 0x4000, 0x3c00, 0x4000, 0x3c00}},
      {0x7c15, 0x4000, 0x3c00, true, {0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00}},
      {0x8000, 0x8000, 0, false, {0x8000, 0, 0, 0x8000, 0}},
      {0, 0x8000, 0x8000, false, {0x8000, 0, 0x8000, 0x8000, 0x8000}},
  };
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned op = 0; op < 5; ++op)
      for (const auto &test : cases)
        for (unsigned selectors = 0; selectors < 16; ++selectors) {
          uint32_t words[4][64];
          uint16_t input[] = {test.a, test.b, test.c};
          for (unsigned r = 0; r < 3; ++r)
            std::fill_n(words[r], 64,
                        (selectors & (1U << r)) ? (uint32_t(input[r]) << 16) | 0x3555U
                                                : 0x35550000U | input[r]);
          const uint32_t *a = words[0], *b = words[1], *c = words[2];
          uint32_t *d = words[3];
          uint64_t flags = cpu | (test.ieee ? GOC_FP_IEEE_MODE : 0);
          uint32_t mode = selectors * GOC_ALU_HIGH_A;
          for (bool wide : {false, true}) {
            std::fill_n(d, 64, 0xdeadbeefU);
            uint32_t state = 0x80000055;
            ASSERT_EQ(
                wide ? wave64[9 + op](flags, 0x55555555aaaaaaaaULL, mode, &d, &a, &b, &c, &state)
                     : legacy[op](flags, 0xaaaaaaaaU, mode, &d, &a, &b, &c, &state),
                GOC_SUCCESS);
            EXPECT_EQ(state, 0x80000055U);
            for (unsigned lane = 0; lane < 64; ++lane) {
              bool active = lane < (wide ? 64U : 32U) && ((0x55555555aaaaaaaaULL >> lane) & 1);
              uint32_t want = (selectors & 8) ? (uint32_t(test.expected[op]) << 16) | 0xbeefU
                                              : 0xdead0000U | test.expected[op];
              EXPECT_EQ(d[lane], active ? want : 0xdeadbeefU)
                  << cpu << "/" << op << "/" << test.ieee << "/" << selectors;
            }
          }
        }
  // RDNA4 propagating selection must see C's signaling NaN even when A/B
  // already produced a quiet NaN. F16C must not quiet C before selection.
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (auto fn : {goc_v_minimum3_f16, goc_v_maximum3_f16, goc_v_minimummaximum_f16,
                    goc_v_maximumminimum_f16}) {
      uint32_t a[32], b[32], c[32], d[32];
      std::fill_n(a, 32, 0x7e12U);
      std::fill_n(b, 32, 0x3c00U);
      std::fill_n(c, 32, 0xfc34U);
      std::fill_n(d, 32, 0xdead0000U);
      const uint32_t *pa = a, *pb = b, *pc = c;
      uint32_t *pd = d;
      ASSERT_EQ(fn(cpu, UINT32_MAX, 0, &pd, &pa, &pb, &pc, nullptr), GOC_SUCCESS);
      for (uint32_t value : d)
        EXPECT_EQ(value, 0xdeadfe34U);
    }
}

TEST(HalfMinmax3, LegacyEveryHalfEncodingPreservedAgainstQuietNaNs) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (auto fn : {goc_v_min3_f16, goc_v_max3_f16, goc_v_med3_f16})
      for (unsigned base = 0; base < 65536; base += 32) {
        uint32_t a[32], b[32], c[32], d[32];
        for (unsigned lane = 0; lane < 32; ++lane) {
          a[lane] = base + lane;
          b[lane] = 0x7e11;
          c[lane] = 0xfe22;
          d[lane] = 0xdead0000;
        }
        const uint32_t *pa = a, *pb = b, *pc = c;
        uint32_t *pd = d;
        ASSERT_EQ(fn(cpu, UINT32_MAX, 0, &pd, &pa, &pb, &pc, nullptr), GOC_SUCCESS);
        for (unsigned lane = 0; lane < 32; ++lane)
          EXPECT_EQ(d[lane], 0xdead0000U | a[lane]) << cpu << "/" << base + lane;
      }
}

TEST(HalfMinmax3, GuestDenormalsAndArchitectureScaling) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (bool rdna3 : {false, true})
      for (unsigned settings = 0; settings < 8; ++settings) {
        uint64_t flags = cpu | ((settings & 1) ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                         ((settings & 2) ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) |
                         ((settings & 4) ? GOC_FP_IEEE_MODE : 0);
        for (uint32_t mode : {0U, GOC_ALU_OMOD_2})
          for (uint32_t input : {1U, 0x3400U}) {
            uint32_t a[64], b[64], c[64], d[64];
            std::fill_n(a, 64, input);
            std::fill_n(b, 64, 0x3800U);
            std::fill_n(c, 64, 0x3c00U);
            std::fill_n(d, 64, 0xdead0000U);
            const uint32_t *pa = a, *pb = b, *pc = c;
            uint32_t *pd = d;
            auto fn = rdna3 ? goc_v_min3_f16_wave64 : goc_v_min3_num_f16_wave64;
            ASSERT_EQ(fn(flags, UINT64_MAX, mode, &pd, &pa, &pb, &pc, nullptr), GOC_SUCCESS);
            bool scale = mode && (!rdna3 || ((settings & 2) && !(settings & 4)));
            uint32_t want =
                input == 1 ? (((settings & 1) || scale) ? 0U : 1U) : (scale ? 0x3800U : 0x3400U);
            for (uint32_t value : d)
              EXPECT_EQ(value, 0xdead0000U | want) << cpu << "/" << settings << "/" << rdna3;
          }
      }
}

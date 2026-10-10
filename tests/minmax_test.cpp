// SPDX-License-Identifier: MIT

#include "dpp_reference.h"
#include "exec_masks.h"
#include "goc/goc.h"
#include "internal.h"
#include "minmax_reference.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {

const decltype(&goc_v_min3_num_f32_wave64) wave64[] = {
    goc_v_min3_num_f32_wave64,       goc_v_max3_num_f32_wave64,
    goc_v_minmax_num_f32_wave64,     goc_v_maxmin_num_f32_wave64,
    goc_v_minimum3_f32_wave64,       goc_v_maximum3_f32_wave64,
    goc_v_minimummaximum_f32_wave64, goc_v_maximumminimum_f32_wave64,
    goc_v_med3_num_f32_wave64,       goc_v_min3_f32_wave64,
    goc_v_max3_f32_wave64,           goc_v_minmax_f32_wave64,
    goc_v_maxmin_f32_wave64,         goc_v_med3_f32_wave64};

using Fn = decltype(&goc_v_fma_f32);
const Fn functions[] = {goc_v_min3_num_f32,       goc_v_max3_num_f32,       goc_v_minmax_num_f32,
                        goc_v_maxmin_num_f32,     goc_v_minimum3_f32,       goc_v_maximum3_f32,
                        goc_v_minimummaximum_f32, goc_v_maximumminimum_f32, goc_v_med3_num_f32};

} // namespace

TEST(Minmax3, AllModifiersMasksAliasesAndSpecialValues) {
  const uint32_t values[] = {0,          0x80000000, 1,          0x80000001, 0x007fffff, 0x00800000,
                             0x3f000000, 0xbf000000, 0x3f800000, 0xbf800000, 0x3f800001, 0x3f7fffff,
                             0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc12345, 0xffc12345,
                             0x3e800000, 0x40000000, 0xc0400000, 0x7f812345, 0xff812346};
  for (int op = 0; op < 9; ++op)
    for (uint32_t mode = 0; mode < 512; ++mode) {
      uint32_t source[3][32], expected[32];
      for (int lane = 0; lane < 32; ++lane) {
        for (int reg = 0; reg < 3; ++reg)
          source[reg][lane] = values[(lane * (2 * reg + 1) + reg * 3) % 23];
        expected[lane] = goc_test::minmax_reference::reference(op, source[0][lane], source[1][lane],
                                                               source[2][lane], mode);
      }
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (uint32_t exec_mask : exec_masks())
          for (int alias = 0; alias < 4; ++alias) {
            SCOPED_TRACE(::testing::Message()
                         << op << "/" << mode << "/" << cpu << "/" << exec_mask << "/" << alias);
            uint32_t storage[4][34], before[32];
            uint32_t *v[4];
            for (int reg = 0; reg < 4; ++reg) {
              std::fill(storage[reg], storage[reg] + 34, 0xdeadbeef);
              v[reg] = storage[reg] + 1;
              if (reg < 3)
                std::copy(source[reg], source[reg] + 32, v[reg]);
            }
            std::copy(v[alias], v[alias] + 32, before);
            ASSERT_EQ(functions[op](cpu, exec_mask, mode, &v[alias], &v[0], &v[1], &v[2], nullptr),
                      GOC_SUCCESS);
            for (int lane = 0; lane < 32; ++lane) {
              uint32_t want = (exec_mask >> lane) & 1 ? expected[lane] : before[lane];
              EXPECT_EQ(v[alias][lane], want);
            }
            for (const auto &reg : storage) {
              EXPECT_EQ(reg[0], 0xdeadbeef);
              EXPECT_EQ(reg[33], 0xdeadbeef);
            }
          }
    }
}

TEST(Minmax3, LiteralOrderNaNsAndSignedZeros) {
  // Inputs A/B/C followed by all eight results in public-function order.
  const uint32_t cases[][11] = {
      {0x3f800000, 0x40000000, 0x40400000, 0x3f800000, 0x40400000, 0x40400000, 0x40000000,
       0x3f800000, 0x40400000, 0x40400000, 0x40000000},
      {0x40400000, 0x40000000, 0x3f800000, 0x3f800000, 0x40400000, 0x40000000, 0x3f800000,
       0x3f800000, 0x40400000, 0x40000000, 0x3f800000},
      {0x80000000, 0, 0x80000000, 0x80000000, 0, 0x80000000, 0x80000000, 0x80000000, 0, 0x80000000,
       0x80000000},
      {0, 0x80000000, 0, 0x80000000, 0, 0, 0, 0x80000000, 0, 0, 0},
      {0x7f800001, 0xff800002, 0x7f800003, 0x7fc00001, 0x7fc00001, 0x7fc00001, 0x7fc00001,
       0x7fc00003, 0x7fc00003, 0x7fc00003, 0x7fc00003},
      {0x7fc00001, 0xff800002, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000,
       0xffc00002, 0xffc00002, 0xffc00002, 0xffc00002},
      {0x7fc00001, 0x40000000, 0x40400000, 0x40000000, 0x40400000, 0x40400000, 0x40000000,
       0x7fc00001, 0x7fc00001, 0x7fc00001, 0x7fc00001},
  };
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (int op = 0; op < 8; ++op)
      for (const auto &test : cases) {
        SCOPED_TRACE(::testing::Message()
                     << cpu << "/" << op << "/" << test[0] << "/" << test[1] << "/" << test[2]);
        uint32_t a[32], b[32], c[32], d[32];
        std::fill(a, a + 32, test[0]);
        std::fill(b, b + 32, test[1]);
        std::fill(c, c + 32, test[2]);
        auto pa = a, pb = b, pc = c, pd = d;
        ASSERT_EQ(functions[op](cpu, UINT32_MAX, 0, &pd, &pa, &pb, &pc, nullptr), GOC_SUCCESS);
        for (uint32_t value : d)
          EXPECT_EQ(value, test[op + 3]);
      }
}

TEST(Minmax3, RandomBitPatterns) {
  std::mt19937 random(20261008);
  for (int batch = 0; batch < 128; ++batch) {
    uint32_t a[32], b[32], c[32], d[32];
    for (int lane = 0; lane < 32; ++lane) {
      a[lane] = random();
      b[lane] = random();
      c[lane] = random();
    }
    auto pa = a, pb = b, pc = c, pd = d;
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int op = 0; op < 9; ++op) {
        SCOPED_TRACE(::testing::Message() << batch << "/" << cpu << "/" << op);
        ASSERT_EQ(functions[op](cpu, UINT32_MAX, 0, &pd, &pa, &pb, &pc, nullptr), GOC_SUCCESS);
        for (int lane = 0; lane < 32; ++lane)
          EXPECT_EQ(d[lane],
                    goc_test::minmax_reference::reference(op, a[lane], b[lane], c[lane], 0));
      }
  }
}

TEST(Minmax3, Validation) {
  for (auto fn : functions) {
    uint32_t a[32] = {}, d[32];
    std::fill(d, d + 32, 0xdeadbeef);
    auto pa = a, pd = d;
    EXPECT_EQ(fn(0, UINT32_MAX, GOC_ALU_HIGH_C, &pd, &pa, &pa, &pa, nullptr),
              GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(fn(0, 0, GOC_ALU_HIGH_D, &pd, &pa, &pa, &pa, nullptr), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(
        fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0, &pd, &pa, &pa, &pa, nullptr),
        GOC_ERROR_UNSUPPORTED_SEMANTICS);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL, 0U, 0x1ff, &pd, &pa, &pa, &pa, nullptr),
              GOC_SUCCESS);
    for (uint32_t value : d)
      EXPECT_EQ(value, 0xdeadbeef);
  }
}

TEST(Minmax3, MedianLiteralNaNsAndZeroTies) {
  const uint32_t cases[][4] = {
      {0x40400000, 0x3f800000, 0x40000000, 0x40000000},
      {0x7f800000, 0xff800000, 0x3f800000, 0x3f800000},
      {0x7fc12345, 0x40000000, 0x3f800000, 0x3f800000},
      {0x40000000, 0x7f812345, 0x3f800000, 0x3f800000},
      {0x40000000, 0x3f800000, 0xff812345, 0x3f800000},
      {0xff812345, 0x7fc12346, 0x7f812347, 0xffc12345},
      {0, 0x80000000, 0x80000000, 0x80000000},
      {0x80000000, 0, 0x80000000, 0},
      {0x80000000, 0x80000000, 0, 0},
      {0x80000000, 0x80000000, 0x80000000, 0x80000000},
      {0, 0, 0x80000000, 0},
      {0x3f800000, 0, 0x80000000, 0},
  };
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (const auto &test : cases) {
      uint32_t a[32], b[32], c[32], d[32];
      std::fill(a, a + 32, test[0]);
      std::fill(b, b + 32, test[1]);
      std::fill(c, c + 32, test[2]);
      auto pa = a, pb = b, pc = c, pd = d;
      ASSERT_EQ(goc_v_med3_num_f32(cpu, UINT32_MAX, 0, &pd, &pa, &pb, &pc, nullptr), GOC_SUCCESS);
      for (uint32_t value : d)
        EXPECT_EQ(value, test[3]) << cpu << "/" << test[0] << "/" << test[1] << "/" << test[2];
    }
}

TEST(Minmax3, Wave64ModifiersDppSelectorsAndAliases) {
  const uint32_t values[] = {0,          0x80000000, 1,          0x80000001, 0x007fffff, 0x00800000,
                             0x3f000000, 0xbf000000, 0x3f800000, 0xbf800000, 0x3f800001, 0x3f7fffff,
                             0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc12345, 0xffc12345,
                             0x3e800000, 0x40000000, 0xc0400000, 0x7f812345, 0xff812346};
  const unsigned reference_ops[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 8};
  for (unsigned op = 0; op < 14; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint32_t mode = 0; mode < 512; ++mode) {
        unsigned route_index = mode % 8;
        uint64_t descriptor = route_index ? goc_test::dpp_modes[route_index - 1] : 0;
        if (descriptor & GOC_DPP16)
          descriptor =
              (descriptor & ~GOC_DPP_ROW_MASK) | (uint64_t((mode >> 3) & 15) << GOC_DPP_ROW_SHIFT);
        uint32_t words[4][66], before[4][66];
        for (unsigned reg = 0; reg < 4; ++reg)
          for (unsigned lane = 0; lane < 66; ++lane)
            words[reg][lane] = values[(lane * (2 * reg + 1) + reg * 3) % 23];
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
            EXPECT_EQ(d[lane], goc_test::minmax_reference::reference(
                                   reference_ops[op], source < 0 ? 0 : before[0][source + 1],
                                   before[1][lane + 1], before[2][lane + 1], mode, op >= 9));
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

TEST(Minmax3, Wave64ValidationAndOptionalState) {
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

TEST(Minmax3, Rdna3GuestSelectionPolicies) {
  const Fn legacy[] = {goc_v_min3_f32, goc_v_max3_f32, goc_v_minmax_f32, goc_v_maxmin_f32,
                       goc_v_med3_f32};

  // Literal expectations from the RDNA3 nested binary selection pseudocode.
  struct Case {
    uint32_t a, b, c;
    bool ieee;
    uint32_t expected[5];
  };

  const Case cases[] = {
      {0x7f812345,
       0x7fc23456,
       0xffc34567,
       false,
       {0x7f812345, 0x7f812345, 0x7f812345, 0x7f812345, 0x7f812345}},
      {0x7f812345,
       0x7fc23456,
       0xffc34567,
       true,
       {0x7fc12345, 0x7fc12345, 0x7fc12345, 0x7fc12345, 0x7fc12345}},
      {0x3f800000,
       0x40000000,
       0xff812345,
       true,
       {0xffc12345, 0xffc12345, 0xffc12345, 0xffc12345, 0xffc12345}},
      {0x3f800000,
       0x40000000,
       0xff812345,
       false,
       {0x3f800000, 0x40000000, 0x3f800000, 0x40000000, 0x3f800000}},
      {0x7f812345,
       0x40000000,
       0x3f800000,
       true,
       {0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000}},
  };
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned op = 0; op < 5; ++op)
      for (const auto &test : cases) {
        uint32_t a[64], b[64], c[64], d[64];
        std::fill_n(a, 64, test.a);
        std::fill_n(b, 64, test.b);
        std::fill_n(c, 64, test.c);
        const uint32_t *pa = a, *pb = b, *pc = c;
        uint32_t *pd = d;
        uint64_t flags = cpu | (test.ieee ? GOC_FP_IEEE_MODE : 0);
        for (bool wide : {false, true}) {
          std::fill_n(d, 64, 0xdeadbeef);
          uint32_t state = 0x80000055;
          ASSERT_EQ(
              wide ? wave64[9 + op](flags, 0x55555555aaaaaaaaULL, 0, &pd, &pa, &pb, &pc, &state)
                   : legacy[op](flags, 0xaaaaaaaaU, 0, &pd, &pa, &pb, &pc, &state),
              GOC_SUCCESS);
          EXPECT_EQ(state, 0x80000055U);
          for (unsigned lane = 0; lane < 64; ++lane) {
            bool active = lane < (wide ? 64U : 32U) && ((0x55555555aaaaaaaaULL >> lane) & 1);
            EXPECT_EQ(d[lane], active ? test.expected[op] : 0xdeadbeefU)
                << cpu << "/" << op << "/" << test.ieee;
          }
        }
      }
}

TEST(Minmax3, GuestDenormalsAndArchitectureScaling) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (bool rdna3 : {false, true})
      for (unsigned settings = 0; settings < 8; ++settings) {
        uint64_t flags = cpu | ((settings & 1) ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                         ((settings & 2) ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) |
                         ((settings & 4) ? GOC_FP_IEEE_MODE : 0);
        uint32_t a[32], b[32], c[32], d[32];
        const uint32_t *pa = a, *pb = b, *pc = c;
        uint32_t *pd = d;
        auto fn = rdna3 ? goc_v_min3_f32 : goc_v_min3_num_f32;
        for (uint32_t mode : {0U, uint32_t(GOC_ALU_OMOD_2)}) {
          for (uint32_t input : {1U, 0x3e800000U}) {
            std::fill_n(a, 32, input);
            std::fill_n(b, 32, 0x3f000000U);
            std::fill_n(c, 32, 0x3f800000U);
            ASSERT_EQ(fn(flags, UINT32_MAX, mode, &pd, &pa, &pb, &pc, nullptr), GOC_SUCCESS);
            bool scale = mode && (!rdna3 || ((settings & 2) && !(settings & 4)));
            uint32_t expected = input == 1 ? (((settings & 1) || scale) ? 0U : 1U)
                                           : (scale ? 0x3f000000U : 0x3e800000U);
            for (uint32_t value : d)
              EXPECT_EQ(value, expected) << cpu << "/" << rdna3 << "/" << settings;
          }
        }
      }
}

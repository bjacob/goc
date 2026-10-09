// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_dpp_reference.h"
#include "rdna4_exec_masks.h"
#include "rdna4_half_reference.h"
#include "rdna4_half_unary_reference.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdint.h>

namespace {

void check(int op, uint32_t actual, uint32_t before, uint16_t want, uint32_t mode) {
  int shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  EXPECT_EQ((actual ^ before) & ~(UINT32_C(0xffff) << shift), 0u);
  uint16_t got = uint16_t(actual >> shift);
  if ((want & 0x7fff) > 0x7c00) {
    EXPECT_GT(got & 0x7fff, 0x7c00);
  } else if (op >= 4 && op <= 8 && (want & 0x7fff) != 0 && (want & 0x7fff) < 0x7c00) {
    // Loose transcendental paths may round at an adjacent half encoding.
    EXPECT_EQ(got & 0x8000, want & 0x8000);
    EXPECT_LE(std::abs(int(got) - int(want)), 1);
  } else {
    EXPECT_EQ(got, want);
  }
}

} // namespace

TEST(HalfUnary, EveryEncodingAndCpuLevel) {
  for (int op = 0; op < 11; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool saturate : {false, true})
        for (uint32_t mode : {UINT32_C(0), GOC_ALU_HIGH_A | GOC_ALU_HIGH_D | GOC_ALU_OMOD_4})
          for (unsigned base = 0; base < 65536; base += 32) {
            SCOPED_TRACE(::testing::Message()
                         << op << '/' << cpu << '/' << saturate << '/' << mode << '/' << base);
            uint32_t input[32], output[32];
            for (int lane = 0; lane < 32; ++lane) {
              auto bits = base + lane;
              input[lane] = mode ? (bits << 16) | 0xbeef : 0xdead0000 | bits;
              output[lane] = 0xdeadbeef;
            }
            auto a = input, d = output;
            ASSERT_EQ(goc_test::half_unary_functions[op](cpu | (saturate ? GOC_FP16_OVFL : 0),
                                                         UINT32_MAX, mode, &d, &a),
                      GOC_SUCCESS);
            for (int lane = 0; lane < 32; ++lane)
              check(op, output[lane], 0xdeadbeef,
                    goc_test::half_unary_reference(op, input[lane], mode, saturate), mode);
          }
}

TEST(HalfUnary, AllModifiersMasksAndAliases) {
  const uint16_t values[] = {0,      0x8000, 1,      0x8001, 0x3ff,  0x400,  0x3800, 0xb800,
                             0x3bff, 0x3c00, 0x3e00, 0xbe00, 0x4100, 0xc100, 0x4200, 0x4400,
                             0x4c00, 0xcc00, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7c01, 0xfe00};
  for (int op = 0; op < 11; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < 128; ++variant)
        for (bool saturate : {false, true})
          for (uint32_t mask : rdna4_exec_masks())
            for (bool alias : {false, true}) {
              auto mode = goc_test::half_unary_modifiers(variant);
              SCOPED_TRACE(::testing::Message() << op << '/' << cpu << '/' << mode << '/'
                                                << saturate << '/' << mask << '/' << alias);
              uint32_t words[2][34], before[2][34];
              for (int reg = 0; reg < 2; ++reg) {
                std::fill(words[reg], words[reg] + 34, 0xdeadbeef);
                for (int lane = 1; lane <= 32; ++lane)
                  words[reg][lane] = values[(lane + reg * 5) % 24] |
                                     (uint32_t(values[(lane * 7 + reg) % 24]) << 16);
              }
              std::memcpy(before, words, sizeof(words));
              auto a = words[0] + 1, d = words[alias ? 0 : 1] + 1;
              ASSERT_EQ(goc_test::half_unary_functions[op](cpu | (saturate ? GOC_FP16_OVFL : 0),
                                                           mask, mode, &d, &a),
                        GOC_SUCCESS);
              for (int reg = 0; reg < 2; ++reg)
                for (int lane = 0; lane < 34; ++lane) {
                  if (reg == (alias ? 0 : 1) && lane >= 1 && lane <= 32 &&
                      ((mask >> (lane - 1)) & 1)) {
                    check(op, words[reg][lane], before[reg][lane],
                          goc_test::half_unary_reference(op, before[0][lane], mode, saturate),
                          mode);
                  } else {
                    EXPECT_EQ(words[reg][lane], before[reg][lane]);
                  }
                }
            }
}

TEST(HalfUnary, LiteralBoundaries) {
  struct Case {
    int op;
    uint16_t input, result;
    uint32_t mode;
    bool saturate;
  };

  const Case cases[] = {{0, 0xbe00, 0xbc00, 0, false},
                        {1, 0xb800, 0x8000, 0, false},
                        {2, 0x3800, 0, 0, false},
                        {2, 0xb800, 0x8000, 0, false},
                        {2, 0x3e00, 0x4000, 0, false},
                        {2, 0x4100, 0x4000, 0, false},
                        {3, 0x8001, 0xbc00, 0, false},
                        {4, 0x4400, 0x4000, 0, false},
                        {5, 0, 0x7c00, 0, true},
                        {5, 1, 0x7bff, 0, true},
                        {5, 1, 0x7c00, 0, false},
                        {5, 0x8000, 0xfc00, 0, true},
                        {6, 0x4400, 0x3800, 0, false},
                        {7, 0x4c00, 0x7c00, 0, false},
                        {7, 0x4c00, 0x7bff, 0, true},
                        {7, 0x7bff, 0x7bff, 0, true},
                        {7, 0x7c00, 0x7c00, 0, true},
                        {7, 0x4c00, 0x7c00, GOC_ALU_OMOD_HALF, false},
                        {7, 0x4c00, 0x77ff, GOC_ALU_OMOD_HALF, true},
                        {8, 0, 0xfbff, 0, true},
                        {8, 0x8000, 0xfbff, 0, true},
                        {8, 0, 0xf7ff, GOC_ALU_OMOD_HALF, true},
                        {7, 0xce00, 1, 0, false},
                        {7, 0xce40, 0, 0, false},
                        {8, 0, 0xfc00, 0, false},
                        {8, 0x4400, 0x4000, 0, false},
                        {9, 0x8001, 0x3bff, 0, false},
                        {9, 0x8000, 0, 0, false},
                        {9, 0xbd00, 0x3a00, 0, false},
                        {10, 1, 0x3800, 0, false},
                        {10, 0x3ff, 0x3bfe, 0, false},
                        {10, 0x8000, 0x8000, 0, false},
                        {10, 0xfc00, 0xfc00, 0, false}};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (const auto &test : cases) {
      SCOPED_TRACE(::testing::Message() << test.op << '/' << cpu << '/' << test.input);
      uint32_t input[32], output[32];
      std::fill(input, input + 32, test.input);
      std::fill(output, output + 32, 0xdeadbeef);
      auto a = input, d = output;
      ASSERT_EQ(goc_test::half_unary_functions[test.op](cpu | (test.saturate ? GOC_FP16_OVFL : 0),
                                                        UINT32_MAX, test.mode, &d, &a),
                GOC_SUCCESS);
      for (auto word : output)
        EXPECT_EQ(word, UINT32_C(0xdead0000) | test.result);
    }
}

TEST(HalfUnary, ValidationAndSemantics) {
  for (auto fn : goc_test::half_unary_functions) {
    uint32_t data[32];
    std::fill(data, data + 32, 0xdeadbeef);
    auto p = data;
    for (uint32_t mask : {UINT32_C(0), UINT32_MAX}) {
      for (int bit = 0; bit < 32; ++bit) {
        if (!(goc_test::half_unary_known & (UINT32_C(1) << bit))) {
          EXPECT_EQ(fn(0, mask, UINT32_C(1) << bit, &p, &p), GOC_ERROR_INVALID_FLAGS);
        }
      }
      EXPECT_EQ(fn(UINT64_C(1) << 63, mask, 0, &p, &p), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, mask, 0, &p, &p),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
    }
    for (auto word : data)
      EXPECT_EQ(word, 0xdeadbeef);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, 0, &p, &p), GOC_SUCCESS);
  }
}

TEST(HalfUnary, ExpLogEveryEncodingAndOutputModifier) {
  for (unsigned op : {7u, 8u})
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned omod = 0; omod < 4; ++omod)
        for (bool clamp : {false, true})
          for (bool saturate : {false, true})
            for (unsigned base = 0; base < 65536; base += 32) {
              uint32_t mode =
                  GOC_ALU_HIGH_A | GOC_ALU_HIGH_D | (omod << 6) | (clamp ? GOC_ALU_CLAMP : 0);
              SCOPED_TRACE(::testing::Message()
                           << op << '/' << cpu << '/' << mode << '/' << saturate << '/' << base);
              uint32_t a[32], d[32];
              for (unsigned lane = 0; lane < 32; ++lane) {
                a[lane] = ((base + lane) << 16) | 0xdead;
                d[lane] = 0xfacecafe;
              }
              auto pa = a, pd = d;
              ASSERT_EQ(goc_test::half_unary_functions[op](cpu | (saturate ? GOC_FP16_OVFL : 0),
                                                           UINT32_MAX, mode, &pd, &pa),
                        GOC_SUCCESS);
              for (unsigned lane = 0; lane < 32; ++lane)
                check(op, d[lane], 0xfacecafe,
                      goc_test::half_unary_reference(op, a[lane], mode, saturate), mode);
            }
}

TEST(HalfUnary, HardwareExpLogRoundingAndOverflow) {
  const uint32_t values[] = {0x4c004bff, 0x00008000, 0x7c00fc00, 0x7bfffbff,
                             0xce40ce00, 0x00018001, 0x3bff3c01, 0x40003c00};
  const uint32_t modes[] = {0, 1, 8, 9, 64, 128, 192, 256, 512, 4096, 4608, 5065, 576, 640, 704};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (bool saturate : {false, true}) {
      uint64_t hash = UINT64_C(14695981039346656037);
      for (uint32_t mask :
           {0xffffffffu, 0u, 0xaaaaaaaau, 0x55555555u, 1u, 0x80000000u, 0xffffu, 0xffff0000u})
        for (unsigned op : {7u, 8u})
          for (uint64_t descriptor : goc_test::dpp_modes)
            for (uint32_t mode : modes) {
              uint32_t a[32], d[32];
              for (unsigned lane = 0; lane < 32; ++lane) {
                a[lane] = values[lane % 8];
                d[lane] = 0xdead0000u + lane;
              }
              auto pa = a, pd = d;
              ASSERT_EQ(goc_test::half_unary_functions[op](cpu | (saturate ? GOC_FP16_OVFL : 0),
                                                           mask, descriptor | mode, &pd, &pa),
                        GOC_SUCCESS);
              unsigned shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
              for (unsigned lane = 0; lane < 32; ++lane) {
                int source;
                uint32_t want = 0xdead0000u + lane;
                if (goc_test::dpp_source(descriptor | mode, mask, lane, source)) {
                  uint32_t value = goc_test::half_unary_reference(op, source < 0 ? 0 : a[source],
                                                                  mode, saturate);
                  want = (want & ~(UINT32_C(65535) << shift)) | (value << shift);
                }
                check(op, d[lane], 0xdead0000u + lane, uint16_t(want >> shift), mode);
                uint32_t word = d[lane];
                if (((word >> shift) & 0x7fff) > 0x7c00)
                  word = (word & ~(UINT32_C(65535) << shift)) | (UINT32_C(0x7e00) << shift);
                hash = (hash ^ word) * UINT64_C(1099511628211);
              }
            }
      EXPECT_EQ(hash, saturate ? UINT64_C(0x7e3c62853aa82d6c) : UINT64_C(0xdeda4067db286486));
    }
}

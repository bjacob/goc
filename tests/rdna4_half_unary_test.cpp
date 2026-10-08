// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_exec_masks.h"
#include "rdna4_half_reference.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_rdna4_v_trunc_f16);
const Fn functions[] = {goc_rdna4_v_trunc_f16, goc_rdna4_v_ceil_f16,      goc_rdna4_v_rndne_f16,
                        goc_rdna4_v_floor_f16, goc_rdna4_v_sqrt_f16,      goc_rdna4_v_rcp_f16,
                        goc_rdna4_v_rsq_f16,   goc_rdna4_v_exp_f16,       goc_rdna4_v_log_f16,
                        goc_rdna4_v_fract_f16, goc_rdna4_v_frexp_mant_f16};
const uint32_t known = GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP |
                       GOC_ALU_HIGH_A | GOC_ALU_HIGH_D;

uint16_t reference(int op, uint32_t input, uint32_t mode, bool saturate) {
  double x = goc_test::half_value(uint16_t(input >> (mode & GOC_ALU_HIGH_A ? 16 : 0)));
  if (mode & GOC_ALU_ABS_A)
    x = std::abs(x);
  if (mode & GOC_ALU_NEG_A)
    x = -x;
  double result = goc_test::half_unary(op, x);
  const double scales[] = {1, 2, 4, 0.5};
  result *= scales[(mode >> 6) & 3];
  if (mode & GOC_ALU_CLAMP)
    result = !(result > 0) ? 0 : std::min(result, 1.0);
  return goc_test::half_bits(result, saturate);
}

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

uint32_t modifiers(unsigned variant) {
  return (variant & 1 ? GOC_ALU_NEG_A : 0) | (variant & 2 ? GOC_ALU_ABS_A : 0) |
         ((variant & 28) << 4) | (variant & 32 ? GOC_ALU_HIGH_A : 0) |
         (variant & 64 ? GOC_ALU_HIGH_D : 0);
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
            ASSERT_EQ(functions[op](cpu | (saturate ? GOC_FP16_OVFL : 0), UINT32_MAX, mode, &d, &a),
                      GOC_SUCCESS);
            for (int lane = 0; lane < 32; ++lane)
              check(op, output[lane], 0xdeadbeef, reference(op, input[lane], mode, saturate), mode);
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
          for (uint64_t mask : rdna4_exec_masks())
            for (bool alias : {false, true}) {
              auto mode = modifiers(variant);
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
              ASSERT_EQ(functions[op](cpu | (saturate ? GOC_FP16_OVFL : 0), mask, mode, &d, &a),
                        GOC_SUCCESS);
              for (int reg = 0; reg < 2; ++reg)
                for (int lane = 0; lane < 34; ++lane) {
                  if (reg == (alias ? 0 : 1) && lane >= 1 && lane <= 32 &&
                      ((mask >> (lane - 1)) & 1)) {
                    check(op, words[reg][lane], before[reg][lane],
                          reference(op, before[0][lane], mode, saturate), mode);
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

  const Case cases[] = {
      {0, 0xbe00, 0xbc00, 0, false}, {1, 0xb800, 0x8000, 0, false},
      {2, 0x3800, 0, 0, false},      {2, 0xb800, 0x8000, 0, false},
      {2, 0x3e00, 0x4000, 0, false}, {2, 0x4100, 0x4000, 0, false},
      {3, 0x8001, 0xbc00, 0, false}, {4, 0x4400, 0x4000, 0, false},
      {5, 0, 0x7c00, 0, true},       {5, 1, 0x7bff, 0, true},
      {5, 1, 0x7c00, 0, false},      {5, 0x8000, 0xfc00, 0, true},
      {6, 0x4400, 0x3800, 0, false}, {7, 0x4c00, 0x7c00, 0, false},
      {7, 0x4c00, 0x7bff, 0, true},  {7, 0x7bff, 0x7bff, 0, true},
      {7, 0x7c00, 0x7c00, 0, true},  {7, 0x4c00, 0x7800, GOC_ALU_OMOD_HALF, false},
      {7, 0xce00, 1, 0, false},      {7, 0xce40, 0, 0, false},
      {8, 0, 0xfc00, 0, false},      {8, 0x4400, 0x4000, 0, false},
      {9, 0x8001, 0x3bff, 0, false}, {9, 0x8000, 0, 0, false},
      {9, 0xbd00, 0x3a00, 0, false}, {10, 1, 0x3800, 0, false},
      {10, 0x3ff, 0x3bfe, 0, false}, {10, 0x8000, 0x8000, 0, false},
      {10, 0xfc00, 0xfc00, 0, false}};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (const auto &test : cases) {
      SCOPED_TRACE(::testing::Message() << test.op << '/' << cpu << '/' << test.input);
      uint32_t input[32], output[32];
      std::fill(input, input + 32, test.input);
      std::fill(output, output + 32, 0xdeadbeef);
      auto a = input, d = output;
      ASSERT_EQ(functions[test.op](cpu | (test.saturate ? GOC_FP16_OVFL : 0), UINT32_MAX, test.mode,
                                   &d, &a),
                GOC_SUCCESS);
      for (auto word : output)
        EXPECT_EQ(word, UINT32_C(0xdead0000) | test.result);
    }
}

TEST(HalfUnary, ValidationAndSemantics) {
  for (auto fn : functions) {
    uint32_t data[32];
    std::fill(data, data + 32, 0xdeadbeef);
    auto p = data;
    for (uint64_t mask : {UINT64_C(0), UINT64_C(0xffffffff00000000), UINT64_MAX}) {
      for (int bit = 0; bit < 32; ++bit) {
        if (!(known & (UINT32_C(1) << bit))) {
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

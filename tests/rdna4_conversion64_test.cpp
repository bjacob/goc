// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_conversion64_reference.h"
#include "rdna4_exec_masks.h"

#include <algorithm>
#include <array>
#include <cfenv>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_rdna4_v_cvt_f64_i32);
const Fn functions[] = {goc_rdna4_v_cvt_f64_i32, goc_rdna4_v_cvt_f64_u32, goc_rdna4_v_cvt_i32_f64,
                        goc_rdna4_v_cvt_u32_f64, goc_rdna4_v_cvt_f64_f32, goc_rdna4_v_cvt_f32_f64};

::testing::AssertionResult check(int op, uint64_t flags, uint32_t mode, const uint64_t input[32]) {
  uint32_t source[2][32], output[2][32];
  for (int lane = 0; lane < 32; ++lane) {
    source[0][lane] = uint32_t(input[lane]);
    source[1][lane] = uint32_t(input[lane] >> 32);
  }
  const uint32_t *a[] = {source[0], op < 2 || op == 4 ? nullptr : source[1]};
  uint32_t *d[] = {output[0], goc_test::conversion64_wide_output(op) ? output[1] : nullptr};
  int status = functions[op](flags, UINT32_MAX, mode, d, a);
  if (status != GOC_SUCCESS)
    return ::testing::AssertionFailure() << "status " << status;
  for (int lane = 0; lane < 32; ++lane) {
    uint64_t actual = output[0][lane];
    if (goc_test::conversion64_wide_output(op))
      actual |= uint64_t(output[1][lane]) << 32;
    uint64_t expected = goc_test::conversion64_reference(op, input[lane], mode);
    if (!goc_test::conversion64_equal(op, actual, expected))
      return ::testing::AssertionFailure()
             << op << "/" << flags << "/" << mode << "/" << lane << std::hex << " input "
             << input[lane] << " expected " << expected << " actual " << actual;
  }
  return ::testing::AssertionSuccess();
}

} // namespace

TEST(Conversion64, LiteralRoundingSaturationAndScalingOrder) {
  struct Case {
    int op;
    uint64_t input, expected;
    uint32_t mode;
  };

  const Case cases[] = {{0, 0xffffffff, 0xbff0000000000000, 0},
                        {0, 0x80000000, 0xc1e0000000000000, 0},
                        {1, 0xffffffff, 0x41efffffffe00000, 0},
                        {1, 0xffffffff, 0x41dfffffffe00000, GOC_ALU_OMOD_HALF},
                        {2, 0x41dfffffffe00000, 0x7fffffff, 0},
                        {2, 0x41dfffffffffffff, 0x7fffffff, 0},
                        {2, 0x41e0000000000000, 0x7fffffff, 0},
                        {2, 0xc1e0000000200000, 0x80000000, 0},
                        {3, 0x41efffffffe00000, 0xffffffff, 0},
                        {3, 0x41efffffffffffff, 0xffffffff, 0},
                        {3, 0x41f0000000000000, 0xffffffff, 0},
                        {3, 0xbff8000000000000, 0, 0},
                        {2, 0xbff8000000000000, 0xffffffff, 0},
                        {2, 0x7ff8000000000001, 0, 0},
                        {3, 0x7ff0000000000000, 0xffffffff, 0},
                        {2, 0xfff0000000000000, 0x80000000, 0},
                        {4, 1, 0x36a0000000000000, 0},
                        {4, 0x80000001, 0xb6a0000000000000, 0},
                        {4, 0x7f7fffff, 0x47efffffe0000000, 0},
                        {4, 0x80000000, 0x8000000000000000, 0},
                        {5, 0x3ff0000010000000, 0x3f800000, 0},
                        {5, 0x3ff0000010000001, 0x3f800001, 0},
                        {5, 0x3ff0000030000000, 0x3f800002, 0},
                        {5, 0x3690000000000000, 0, 0},
                        {5, 0x3690000000000000, 0, GOC_ALU_OMOD_2},
                        {5, 0x3690000000000001, 1, 0},
                        {5, 0xb690000000000000, 0x80000000, 0},
                        {5, 0x36a8000000000000, 2, 0},
                        {5, 0x36a8000000000000, 1, GOC_ALU_OMOD_HALF},
                        {5, 0x47effffff0000000, 0x7f800000, 0},
                        {5, 0x47effffff0000000, 0x7f800000, GOC_ALU_OMOD_HALF},
                        {5, 0x47effffff0000000, 0x3f800000, GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},
                        {5, 0xfff8000000001234, 0, GOC_ALU_CLAMP}};
  for (const auto &c : cases)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t low[32], high[32], output[2][32];
      std::fill_n(low, 32, uint32_t(c.input));
      std::fill_n(high, 32, uint32_t(c.input >> 32));
      const uint32_t *a[] = {low, high};
      uint32_t *d[] = {output[0], output[1]};
      ASSERT_EQ(functions[c.op](cpu, UINT32_MAX, c.mode, d, a), GOC_SUCCESS);
      for (int lane = 0; lane < 32; ++lane) {
        uint64_t actual = output[0][lane];
        if (goc_test::conversion64_wide_output(c.op))
          actual |= uint64_t(output[1][lane]) << 32;
        ASSERT_EQ(actual, c.expected) << c.op << "/" << cpu << "/" << std::hex << c.input;
      }
    }
}

TEST(Conversion64, ExponentBoundariesAndEveryModifier) {
  for (int op = 0; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < goc_test::conversion64_modes(op); ++variant) {
        unsigned count = op < 2 ? 32 : op == 4 ? 256 : 2048;
        for (unsigned exponent = 0; exponent < count; ++exponent) {
          uint64_t input[32];
          for (unsigned lane = 0; lane < 32; ++lane) {
            uint64_t base =
                op < 2 ? UINT64_C(1) << exponent : uint64_t(exponent) << (op == 4 ? 23 : 52);
            input[lane] =
                (base + lane % 16 - 8) ^ (uint64_t(lane / 16) << (op < 2 || op == 4 ? 31 : 63));
            if (op < 2 || op == 4)
              input[lane] = uint32_t(input[lane]);
          }
          ASSERT_TRUE(check(op, cpu, goc_test::conversion64_mode(op, variant), input));
        }
      }
}

TEST(Conversion64, NarrowingMidpointsAcrossTheFp32Range) {
  const uint64_t fractions[] = {0, 1, 2, 0x3fffff, 0x7ffffe, 0x7fffff};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned variant = 0; variant < 32; ++variant)
      for (unsigned exponent = 0; exponent < 255; ++exponent)
        for (auto fraction : fractions) {
          uint64_t significand = 2 * ((exponent ? 0x800000 : 0) | fraction) + 1;
          uint64_t midpoint = goc_test::conversion_encode(
              false, significand, (exponent ? int(exponent) - 127 : -126) - 24, 52, 1023);
          uint64_t input[32];
          for (unsigned lane = 0; lane < 32; ++lane)
            input[lane] = (midpoint + lane % 3 - 1) | (uint64_t(lane % 2) << 63);
          ASSERT_TRUE(check(5, cpu, goc_test::conversion64_mode(5, variant), input));
        }
}

TEST(Conversion64, RandomFullWordsAndEveryModifier) {
  for (int op = 0; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < goc_test::conversion64_modes(op); ++variant) {
        std::mt19937 random(684);
        for (int batch = 0; batch < 256; ++batch) {
          uint64_t input[32];
          for (auto &word : input) {
            word = (uint64_t(random()) << 32) | random();
            if (op < 2 || op == 4)
              word = uint32_t(word);
          }
          ASSERT_TRUE(check(op, cpu, goc_test::conversion64_mode(op, variant), input));
        }
      }
}

TEST(Conversion64, MasksCrossHalfAliasesAndUnalignedStorage) {
  std::array<std::array<int, 4>, 32> layouts;
  for (int i = 0; i < 32; ++i)
    layouts[i] = {0, i / 16, (i / 4) % 4, i % 4};
  std::mt19937 random(9673);
  for (int op = 0; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < goc_test::conversion64_modes(op); ++variant)
        for (uint64_t mask : rdna4_exec_masks())
          for (const auto &layout : layouts) {
            uint32_t storage[4][34], expected[4][34];
            for (int reg = 0; reg < 4; ++reg)
              for (int word = 0; word < 34; ++word)
                storage[reg][word] = expected[reg][word] = random();
            const uint32_t *a[] = {storage[layout[0]] + 1, storage[layout[1]] + 1};
            uint32_t *d[] = {storage[layout[2]] + 1, storage[layout[3]] + 1};
            uint32_t mode = goc_test::conversion64_mode(op, variant);
            uint64_t results[32];
            for (int lane = 0; lane < 32; ++lane) {
              uint64_t raw = a[0][lane];
              if (op >= 2 && op != 4)
                raw |= uint64_t(a[1][lane]) << 32;
              results[lane] = goc_test::conversion64_reference(op, raw, mode);
            }
            ASSERT_EQ(functions[op](cpu, mask, mode, d, a), GOC_SUCCESS);
            // Loose NaN payloads are unspecified. Check quiet-NaN classification
            // even when D0 == D1 leaves only the high word, then use the actual
            // payload for the whole-buffer preservation checks.
            uint64_t nan = goc_test::conversion64_wide_output(op) ? UINT64_C(0x7ff8000000000000)
                                                                  : UINT64_C(0x7fc00000);
            for (int lane = 0; lane < 32; ++lane)
              if (((mask >> lane) & 1) && op != 2 && op != 3 &&
                  goc_test::conversion64_equal(op, results[lane], nan)) {
                uint64_t actual = d[0][lane];
                if (goc_test::conversion64_wide_output(op))
                  actual = (layout[2] == layout[3] ? 0 : actual) | (uint64_t(d[1][lane]) << 32);
                ASSERT_TRUE(goc_test::conversion64_equal(op, actual, nan));
                results[lane] = actual;
              }
            for (int reg = 0; reg < (goc_test::conversion64_wide_output(op) ? 2 : 1); ++reg)
              for (int lane = 0; lane < 32; ++lane)
                if ((mask >> lane) & 1)
                  expected[layout[2 + reg]][lane + 1] = uint32_t(results[lane] >> (32 * reg));
            for (int reg = 0; reg < 4; ++reg)
              for (int word = 0; word < 34; ++word)
                ASSERT_EQ(storage[reg][word], expected[reg][word])
                    << op << "/" << cpu << "/" << mode;
          }
}

TEST(Conversion64, ValidationAndSemanticFallback) {
  for (int op = 0; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t source[2][32] = {}, output[2][32];
      uint64_t input[32] = {};
      const uint32_t *a[] = {source[0], source[1]};
      uint32_t *d[] = {output[0], output[1]};
      std::fill_n(output[0], 32, 0xdeadbeef);
      std::fill_n(output[1], 32, 0xdeadbeef);
      uint32_t known =
          GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP | (op < 2 ? 0 : GOC_ALU_NEG_A | GOC_ALU_ABS_A);
      for (unsigned bit = 0; bit < 32; ++bit)
        if (!(known & (uint32_t(1) << bit))) {
          EXPECT_EQ(functions[op](cpu, UINT32_MAX, uint32_t(1) << bit, d, a),
                    GOC_ERROR_INVALID_FLAGS);
          EXPECT_EQ(functions[op](cpu, 0, uint32_t(1) << bit, nullptr, nullptr),
                    GOC_ERROR_INVALID_FLAGS);
        }
      EXPECT_EQ(functions[op](cpu | (UINT64_C(1) << 63), UINT32_MAX, 0, d, a),
                GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(functions[op](cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                              UINT32_MAX, 0, d, a),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
      for (auto &reg : output)
        for (auto word : reg)
          EXPECT_EQ(word, 0xdeadbeef);
      EXPECT_EQ(functions[op](cpu, UINT64_C(0xffffffff00000000), 0, nullptr, nullptr), GOC_SUCCESS);
      for (unsigned sem = 1; sem < 4; ++sem)
        ASSERT_TRUE(check(op, cpu | (uint64_t(sem) << 16), 0, input));
    }
}

TEST(Conversion64, IntegerOutputsIgnoreHostRounding) {
  fenv_t original;
  ASSERT_EQ(std::fegetenv(&original), 0);
  const uint64_t values[] = {0x3ff8000000000000, 0xbff8000000000000,
                             0x41dfffffffffffff, 0x41efffffffffffff,
                             0xc1e0000000100000, 1,
                             0x8000000000000001, 0};
  uint64_t input[32];
  for (int lane = 0; lane < 32; ++lane)
    input[lane] = values[lane % 8];
  for (int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
    EXPECT_EQ(std::fesetround(rounding), 0);
    for (int op : {2, 3})
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
        EXPECT_TRUE(check(op, cpu, 0, input));
        EXPECT_EQ(std::fegetround(), rounding);
      }
  }
  EXPECT_EQ(std::fesetenv(&original), 0);
}

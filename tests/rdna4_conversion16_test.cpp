// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_conversion16_hardware.h"
#include "rdna4_conversion16_reference.h"
#include "rdna4_conversion64_reference.h"
#include "rdna4_dpp_reference.h"
#include "rdna4_exec_masks.h"

#include <algorithm>
#include <cfenv>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_rdna4_v_cvt_f16_i16);
const Fn functions[] = {goc_rdna4_v_cvt_f16_i16, goc_rdna4_v_cvt_f16_u16, goc_rdna4_v_cvt_i16_f16,
                        goc_rdna4_v_cvt_u16_f16, goc_rdna4_v_cvt_f16_f32, goc_rdna4_v_cvt_f32_f16};

::testing::AssertionResult check(int op, uint64_t flags, uint32_t mode, const uint32_t input[32]) {
  uint32_t output[32];
  std::fill_n(output, 32, 0xa5a55a5a);
  const uint32_t *a[] = {input};
  uint32_t *d[] = {output};
  int status = functions[op](flags, UINT32_MAX, mode, d, a);
  if (status != GOC_SUCCESS)
    return ::testing::AssertionFailure() << "status " << status;
  for (int lane = 0; lane < 32; ++lane) {
    uint32_t expected =
        goc_test::conversion16_reference(op, input[lane], 0xa5a55a5a, mode, flags & GOC_FP16_OVFL);
    if (!goc_test::conversion16_equal(op, output[lane], expected, mode))
      return ::testing::AssertionFailure()
             << op << "/" << flags << "/" << mode << "/" << lane << std::hex << " input "
             << input[lane] << " expected " << expected << " actual " << output[lane];
  }
  return ::testing::AssertionSuccess();
}

} // namespace

TEST(Conversion16, HardwareCapturedRoundingOverflowAndModifiers) {
  const uint32_t modes[] = {0,
                            GOC_ALU_OMOD_2,
                            GOC_ALU_OMOD_4,
                            GOC_ALU_OMOD_HALF,
                            GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_2 | GOC_ALU_CLAMP,
                            GOC_ALU_CLAMP,
                            GOC_ALU_OMOD_HALF,
                            GOC_ALU_OMOD_2};
  for (const auto &capture : goc_test::half_conversion_captures)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int sat = 0; sat < 2; ++sat)
        for (int column = 0; column < 8; ++column) {
          uint32_t input[32], output[32];
          std::fill_n(input, 32, capture.source);
          std::fill_n(output, 32, 0xa5a5a5a5);
          const uint32_t *a[] = {input};
          uint32_t *d[] = {output};
          int op = column < 6 ? 4 : column == 6 ? 1 : 0;
          ASSERT_EQ(functions[op](cpu | (sat ? GOC_FP16_OVFL : 0), UINT32_MAX, modes[column], d, a),
                    GOC_SUCCESS);
          for (auto actual : output)
            ASSERT_EQ(actual, 0xa5a50000u | capture.expected[sat][column])
                << column << "/" << cpu << "/" << sat << "/" << std::hex << capture.source;
        }
}

TEST(Conversion16, EveryHalfEncodingAndIntegerInput) {
  for (int op : {0, 1, 2, 3, 5})
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool sat : {false, true})
        for (unsigned variant : {0u, goc_test::conversion16_modes(op) - 1})
          for (unsigned start = 0; start < 65536; start += 32) {
            uint32_t input[32];
            for (unsigned lane = 0; lane < 32; ++lane)
              input[lane] = (start + lane) | ((65535 - start - lane) << 16);
            ASSERT_TRUE(check(op, cpu | (sat ? GOC_FP16_OVFL : 0),
                              goc_test::conversion16_mode(op, variant), input));
          }
}

TEST(Conversion16, RandomWordsAndEveryModifier) {
  for (int op = 0; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool sat : {false, true})
        for (unsigned variant = 0; variant < goc_test::conversion16_modes(op); ++variant) {
          std::mt19937 random(461);
          for (int batch = 0; batch < 64; ++batch) {
            uint32_t input[32];
            for (auto &word : input)
              word = random();
            ASSERT_TRUE(check(op, cpu | (sat ? GOC_FP16_OVFL : 0),
                              goc_test::conversion16_mode(op, variant), input));
          }
        }
}

TEST(Conversion16, EveryNarrowingMidpointAndAdjacentFloat) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (bool sat : {false, true})
      for (unsigned omod = 0; omod < 4; ++omod)
        for (unsigned start = 0; start < 0x7c00 * 6; start += 32) {
          uint32_t input[32];
          for (unsigned lane = 0; lane < 32; ++lane) {
            unsigned index = (start + lane) % (0x7c00 * 6), half = index / 6;
            unsigned exponent = half >> 10, fraction = half & 1023;
            uint64_t significand = 2 * ((exponent ? 1024 : 0) | fraction) + 1;
            uint32_t midpoint = uint32_t(goc_test::conversion_encode(
                false, significand, (exponent ? int(exponent) - 15 : -14) - 11, 23, 127));
            input[lane] = (midpoint + index % 3 - 1) ^ (uint32_t((index / 3) % 2) << 31);
          }
          ASSERT_TRUE(check(4, cpu | (sat ? GOC_FP16_OVFL : 0), omod << 6, input));
        }
}

TEST(Conversion16, EveryModifierMasksAliasesAndUnalignedStorage) {
  std::mt19937 random(8773);
  for (int op = 0; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool sat : {false, true})
        for (unsigned variant = 0; variant < goc_test::conversion16_modes(op); ++variant)
          for (uint32_t mask : rdna4_exec_masks())
            for (bool alias : {false, true}) {
              uint32_t storage[2][34], original[2][34];
              for (int reg = 0; reg < 2; ++reg)
                for (int word = 0; word < 34; ++word)
                  storage[reg][word] = original[reg][word] = random();
              uint32_t mode = goc_test::conversion16_mode(op, variant);
              const uint32_t *a[] = {storage[0] + 1};
              uint32_t *d[] = {storage[alias ? 0 : 1] + 1};
              ASSERT_EQ(functions[op](cpu | (sat ? GOC_FP16_OVFL : 0), mask, mode, d, a),
                        GOC_SUCCESS);
              for (int reg = 0; reg < 2; ++reg)
                for (int word = 0; word < 34; ++word) {
                  uint32_t expected = original[reg][word];
                  if (reg == (alias ? 0 : 1) && word > 0 && word <= 32 &&
                      ((mask >> (word - 1)) & 1)) {
                    expected = goc_test::conversion16_reference(op, original[0][word], expected,
                                                                mode, sat);
                    ASSERT_TRUE(
                        goc_test::conversion16_equal(op, storage[reg][word], expected, mode))
                        << op << "/" << cpu << "/" << mode;
                  } else {
                    ASSERT_EQ(storage[reg][word], expected);
                  }
                }
            }
}

TEST(Conversion16, ValidationAndSemanticFallback) {
  for (int op = 0; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t input[32] = {}, output[32];
      std::fill_n(output, 32, 0xdeadbeef);
      const uint32_t *a[] = {input};
      uint32_t *d[] = {output};
      uint32_t known = goc_test::conversion16_mode(op, goc_test::conversion16_modes(op) - 1);
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
      for (auto word : output)
        EXPECT_EQ(word, 0xdeadbeef);
      EXPECT_EQ(functions[op](cpu, UINT32_C(0), 0, nullptr, nullptr), GOC_SUCCESS);
      for (unsigned sem = 1; sem < 4; ++sem)
        ASSERT_TRUE(check(op, cpu | (uint64_t(sem) << 16), 0, input));
    }
}

TEST(Conversion16, IntegerOutputsIgnoreHostRounding) {
  fenv_t original;
  ASSERT_EQ(std::fegetenv(&original), 0);
  uint32_t input[32];
  const uint32_t halves[] = {0x3e00, 0xbe00, 1, 0x8001, 0x7bff, 0xfbff, 0x7c01, 0x7c00};
  for (int lane = 0; lane < 32; ++lane)
    input[lane] = halves[lane % 8];
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

TEST(Conversion16, DppMasksAliasesAndUnalignedStorage) {
  std::mt19937 random(8773);
  for (int op = 0; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint64_t descriptor : goc_test::dpp_modes)
        for (bool sat : {false, true})
          for (unsigned variant :
               {0u, goc_test::conversion16_modes(op) / 2, goc_test::conversion16_modes(op) - 1})
            for (uint32_t mask : rdna4_exec_masks())
              for (bool alias : {false, true}) {
                uint32_t storage[2][34], original[2][34];
                for (int reg = 0; reg < 2; ++reg)
                  for (int word = 0; word < 34; ++word)
                    storage[reg][word] = original[reg][word] = random();
                uint32_t mode = goc_test::conversion16_mode(op, variant);
                const uint32_t *a[] = {storage[0] + 1};
                uint32_t *d[] = {storage[alias ? 0 : 1] + 1};
                ASSERT_EQ(
                    functions[op](cpu | (sat ? GOC_FP16_OVFL : 0), mask, descriptor | mode, d, a),
                    GOC_SUCCESS);
                for (int reg = 0; reg < 2; ++reg)
                  for (int word = 0; word < 34; ++word) {
                    uint32_t expected = original[reg][word];
                    int source;
                    if (reg == (alias ? 0 : 1) && word > 0 && word <= 32 &&
                        goc_test::dpp_source(descriptor, mask, word - 1, source)) {
                      expected = goc_test::conversion16_reference(
                          op, source < 0 ? 0u : original[0][source + 1], expected, mode, sat);
                      ASSERT_TRUE(
                          goc_test::conversion16_equal(op, storage[reg][word], expected, mode))
                          << op << "/" << cpu << "/" << mode;
                    } else {
                      ASSERT_EQ(storage[reg][word], expected);
                    }
                  }
              }
}

TEST(Conversion16, DppEveryModifier) {
  std::mt19937 random(603);
  for (unsigned op = 0; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool saturate : {false, true})
        for (uint64_t descriptor : goc_test::dpp_modes)
          for (unsigned variant = 0; variant < goc_test::conversion16_modes(op); ++variant) {
            uint32_t mode = goc_test::conversion16_mode(op, variant);
            uint32_t a[32], d[32];
            for (unsigned lane = 0; lane < 32; ++lane) {
              a[lane] = random();
              d[lane] = 0xdeadbeef;
            }
            auto pa = a, pd = d;
            ASSERT_EQ(functions[op](cpu | (saturate ? GOC_FP16_OVFL : 0), UINT32_MAX,
                                    descriptor | mode, &pd, &pa),
                      GOC_SUCCESS);
            for (unsigned lane = 0; lane < 32; ++lane) {
              int source;
              if (goc_test::dpp_source(descriptor, UINT32_MAX, lane, source)) {
                uint32_t want = goc_test::conversion16_reference(op, source < 0 ? 0u : a[source],
                                                                 0xdeadbeef, mode, saturate);
                EXPECT_TRUE(goc_test::conversion16_equal(op, d[lane], want, mode));
              } else {
                EXPECT_EQ(d[lane], 0xdeadbeef);
              }
            }
          }
}

TEST(Conversion16, DppHardwareCorpus) {
  const uint32_t values[] = {0x00018001, 0x03ff83ff, 0x04008400, 0x04018401, 0x7bfffbff, 0x7c00fc00,
                             0x7e007c01, 0x80000000, 0x3f800000, 0xbf000000, 0x387fefff, 0x387ff000,
                             0x387ff001, 0x477ff000, 0x7f800001, 0xff800001};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (bool saturate : {false, true})
      for (uint32_t mask :
           {0xffffffffu, 0u, 0xaaaaaaaau, 0x55555555u, 1u, 0x80000000u, 0xffffu, 0xffff0000u})
        for (unsigned op = 0; op < 6; ++op)
          for (uint64_t descriptor : goc_test::dpp_modes)
            for (unsigned variant :
                 {0u, 1u, 2u, 3u, 4u, 7u, 8u, 16u, goc_test::conversion16_modes(op) / 2,
                  goc_test::conversion16_modes(op) - 1}) {
              uint32_t mode = goc_test::conversion16_mode(op, variant);
              uint32_t a[32], d[32];
              for (unsigned lane = 0; lane < 32; ++lane) {
                a[lane] = values[lane % 16];
                d[lane] = 0xdead0000u + lane;
              }
              auto pa = a, pd = d;
              ASSERT_EQ(functions[op](cpu | (saturate ? GOC_FP16_OVFL : 0), mask, descriptor | mode,
                                      &pd, &pa),
                        GOC_SUCCESS);
              for (unsigned lane = 0; lane < 32; ++lane) {
                int source;
                uint32_t want = 0xdead0000u + lane;
                if (goc_test::dpp_source(descriptor, mask, lane, source)) {
                  want = goc_test::conversion16_reference(op, source < 0 ? 0u : a[source], want,
                                                          mode, saturate);
                  EXPECT_TRUE(goc_test::conversion16_equal(op, d[lane], want, mode));
                } else {
                  EXPECT_EQ(d[lane], want);
                }
                uint32_t word = d[lane];
                unsigned shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
                if (op == 5) {
                  if ((word & 0x7fffffff) > 0x7f800000)
                    word = 0x7fc00000;
                } else if (op != 2 && op != 3 && ((word >> shift) & 0x7fff) > 0x7c00) {
                  word = (word & ~(UINT32_C(65535) << shift)) | (UINT32_C(0x7e00) << shift);
                }
                hash = (hash ^ word) * UINT64_C(1099511628211);
              }
            }
    EXPECT_EQ(hash, UINT64_C(0x6e08f00f982b6b95));
  }
}

TEST(Conversion16, DppValidation) {
  for (unsigned op = 0; op < 6; ++op)
    for (uint64_t descriptor : goc_test::dpp_modes) {
      EXPECT_EQ(functions[op](0, 0, descriptor, nullptr, nullptr), GOC_SUCCESS);
      EXPECT_EQ(functions[op](0, UINT32_MAX, descriptor | GOC_ALU_NEG_B, nullptr, nullptr),
                GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(functions[op](0, UINT32_MAX, descriptor | (UINT64_C(1) << 36), nullptr, nullptr),
                GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(functions[op](GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, UINT32_MAX,
                              descriptor, nullptr, nullptr),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
    }
}

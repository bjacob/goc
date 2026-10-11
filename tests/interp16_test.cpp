// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "exec_masks.h"
#include "fp_environment.h"
#include "fuzzy.h"
#include "goc/goc.h"
#include "half_reference.h"
#include "interp16_hardware.h"
#include "interp16_reference.h"
#include "interp32_reference.h"
#include "loose_tolerances.h"

#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_v_interp_p10_f16_f32);
const Fn functions[] = {goc_v_interp_p10_f16_f32, goc_v_interp_p2_f16_f32,
                        goc_v_interp_p10_rtz_f16_f32, goc_v_interp_p2_rtz_f16_f32};

const decltype(&goc_v_interp_p10_f16_f32_wave64) wave64_functions[] = {
    goc_v_interp_p10_f16_f32_wave64, goc_v_interp_p2_f16_f32_wave64,
    goc_v_interp_p10_rtz_f16_f32_wave64, goc_v_interp_p2_rtz_f16_f32_wave64};

void wave64_capture_inputs(uint32_t words[4][64], unsigned start) {
  for (unsigned half = 0; half < 2; ++half) {
    uint32_t captured[4][32];
    goc_test::interp16_capture_inputs(captured, start + half * 32);
    for (unsigned reg = 0; reg < 4; ++reg)
      std::memcpy(words[reg] + half * 32, captured[reg], sizeof(captured[reg]));
  }
}

::testing::AssertionResult compare_result(unsigned op, unsigned m, uint32_t actual,
                                          uint32_t expected) {
  actual = goc_test::interp16_canonical(op, m, actual);
  expected = goc_test::interp16_canonical(op, m, expected);
  if (!(op & 1)) {
    unsigned magnitude = expected & 0x7fffffff;
    if (magnitude && magnitude < 0x7f800000)
      return goc_test::fuzzy(goc_test::interp32_float(actual), goc_test::interp32_float(expected),
                             goc_test::fma_f32_tolerance);
  } else {
    unsigned shift = m & 32 ? 16 : 0;
    if ((actual & ~(0xffffU << shift)) != (expected & ~(0xffffU << shift)))
      return ::testing::AssertionFailure() << "unselected destination half changed";
    uint16_t got = uint16_t(actual >> shift), want = uint16_t(expected >> shift);
    if ((want & 0x7fff) && (want & 0x7fff) < 0x7c00)
      return goc_test::fuzzy(goc_test::half_value(got), goc_test::half_value(want),
                             goc_test::fma_f16_tolerance);
  }
  if (actual == expected)
    return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure()
         << "categorical result: actual=" << actual << " expected=" << expected;
}

} // namespace

TEST(Interp16, HardwareModifiersRoundingAndOverflow) {
  for (unsigned op = 0; op < 4; ++op)
    for (unsigned ovfl = 0; ovfl < 2; ++ovfl)
      for (unsigned m = 0; m < 64; ++m) {
        uint64_t digest = goc_test::capture_hash_seed;
        for (unsigned start = 0; start < 1024; start += 32) {
          uint32_t words[4][32];
          goc_test::interp16_capture_inputs(words, start);
          for (unsigned lane = 0; lane < 32; ++lane) {
            uint32_t expected = goc_test::interp16_canonical(
                op, m,
                goc_test::interp16_reference(op, words[0], words[1], words[2], words[3][lane], lane,
                                             m, ovfl));
            digest = goc_test::capture_hash_bytes(digest, expected, 4);
          }
        }
        ASSERT_EQ(digest, goc_test::interp16_digests[(op * 2 + ovfl) * 64 + m])
            << "reference corpus: " << op << "/" << ovfl << "/" << m;
      }

  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned op = 0; op < 4; ++op)
      for (unsigned ovfl = 0; ovfl < 2; ++ovfl)
        for (unsigned m = 0; m < 64; ++m) {
          uint64_t digest = goc_test::capture_hash_seed;
          for (unsigned start = 0; start < 1024; start += 32) {
            uint32_t words[4][32];
            goc_test::interp16_capture_inputs(words, start);
            const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
            uint32_t *d[] = {words[3]};
            ASSERT_EQ(functions[op](cpu | (ovfl ? GOC_FP16_OVFL : 0), UINT32_MAX,
                                    goc_test::interp16_mode(op, m, 7), d, a, b, c),
                      GOC_SUCCESS);
            for (uint32_t value : words[3]) {
              value = goc_test::interp16_canonical(op, m, value);
              digest = goc_test::capture_hash_bytes(digest, value, 4);
            }
          }
          EXPECT_EQ(digest, goc_test::interp16_digests[(op * 2 + ovfl) * 64 + m])
              << cpu << "/" << op << "/" << ovfl << "/" << m;
        }
}

TEST(Interp16, MasksAliasesUnalignedStorageAndWaits) {
  const unsigned sources[][3] = {{0, 1, 2}, {0, 0, 2}, {0, 1, 0}, {0, 1, 1}, {0, 0, 0}};
  for (unsigned op = 0; op < 4; ++op)
    for (unsigned m = 0; m < 64; ++m)
      for (unsigned ovfl = 0; ovfl < 2; ++ovfl)
        for (const auto &source : sources)
          for (unsigned target = 0; target < 4; ++target) {
            uint32_t initial[4][35], result[32];
            for (unsigned reg = 0; reg < 4; ++reg)
              for (unsigned lane = 0; lane < 35; ++lane)
                initial[reg][lane] = (lane * 0x7395a831u) ^ (reg * 0xa7925163u);
            for (unsigned lane = 0; lane < 32; ++lane)
              result[lane] = goc_test::interp16_reference(
                  op, initial[source[0]] + 1, initial[source[1]] + 1, initial[source[2]] + 1,
                  initial[target][lane + 1], lane, m, ovfl);
            for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
              for (uint32_t exec_mask : exec_masks()) {
                uint32_t words[4][35];
                std::memcpy(words, initial, sizeof(words));
                const uint32_t *a[] = {words[source[0]] + 1}, *b[] = {words[source[1]] + 1},
                               *c[] = {words[source[2]] + 1};
                uint32_t *d[] = {words[target] + 1};
                ASSERT_EQ(functions[op](cpu | (ovfl ? GOC_FP16_OVFL : 0), exec_mask,
                                        goc_test::interp16_mode(op, m, m % 8), d, a, b, c),
                          GOC_SUCCESS);
                for (unsigned reg = 0; reg < 4; ++reg)
                  for (unsigned lane = 0; lane < 35; ++lane) {
                    uint32_t want = initial[reg][lane], got = words[reg][lane];
                    if (reg == target && lane > 0 && lane <= 32 &&
                        ((exec_mask >> (lane - 1)) & 1)) {
                      ASSERT_TRUE(compare_result(op, m, got, result[lane - 1]))
                          << cpu << "/" << op << "/" << m << "/" << target << "/" << lane;
                      continue;
                    }
                    ASSERT_EQ(got, want)
                        << cpu << "/" << op << "/" << m << "/" << target << "/" << lane;
                  }
              }
          }
}

TEST(Interp16, IndependentIntegerOracleAndRoundingBoundaries) {
  const uint32_t edges[] = {0,          0x80000000, 1,          0x3f800000, 0x3f800001, 0x3f800002,
                            0xbf800001, 0x3a000000, 0x33800000, 0x7f7fffff, 0xff7fffff, 0x00800000,
                            0x00013c00, 0x3c010001, 0x7bfffbff, 0x7c007e00};
  for (unsigned op = 0; op < 4; ++op)
    for (unsigned m = 0; m < 64; ++m)
      for (unsigned ovfl = 0; ovfl < 2; ++ovfl) {
        std::mt19937 random(314159);
        for (unsigned batch = 0; batch < 24; ++batch) {
          uint32_t initial[4][32], expected[32];
          for (unsigned reg = 0; reg < 4; ++reg)
            for (unsigned lane = 0; lane < 32; ++lane)
              initial[reg][lane] =
                  batch < 16 ? edges[(batch + reg * 7 + lane) % 16] : uint32_t(random());
          for (unsigned lane = 0; lane < 32; ++lane)
            expected[lane] = goc_test::interp16_canonical(
                op, m,
                goc_test::interp16_reference(op, initial[0], initial[1], initial[2],
                                             initial[3][lane], lane, m, ovfl));
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
            uint32_t words[4][32];
            std::memcpy(words, initial, sizeof(words));
            const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
            uint32_t *d[] = {words[3]};
            ASSERT_EQ(functions[op](cpu | (ovfl ? GOC_FP16_OVFL : 0), UINT32_MAX,
                                    goc_test::interp16_mode(op, m), d, a, b, c),
                      GOC_SUCCESS);
            ASSERT_EQ(std::memcmp(words, initial, sizeof(initial[0]) * 3), 0);
            for (unsigned lane = 0; lane < 32; ++lane)
              ASSERT_TRUE(compare_result(op, m, words[3][lane], expected[lane]))
                  << cpu << "/" << op << "/" << m << "/" << ovfl << "/" << batch << "/" << lane;
          }
        }
      }
}

TEST(Interp16, ValidationAndUnchangedHostRounding) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  std::fesetround(FE_TONEAREST);
  for (unsigned op = 0; op < 4; ++op) {
    uint32_t known = goc_test::interp16_mode(op, 63, 7);
    EXPECT_EQ(functions[op](0, 0U, known, nullptr, nullptr, nullptr, nullptr), GOC_SUCCESS);
    for (unsigned bit = 0; bit < 32; ++bit) {
      if (!(known & (1u << bit))) {
        EXPECT_EQ(functions[op](0, 0, 1u << bit, nullptr, nullptr, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
      }
    }
    EXPECT_EQ(functions[op](GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0, nullptr,
                            nullptr, nullptr, nullptr),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t words[4][32];
      goc_test::interp16_capture_inputs(words, 0);
      const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
      uint32_t *d[] = {words[3]};
      std::feraiseexcept(FE_DIVBYZERO);
      EXPECT_EQ(functions[op](cpu, UINT32_MAX, known, d, a, b, c), GOC_SUCCESS);
      EXPECT_EQ(std::fegetround(), FE_TONEAREST);
      EXPECT_TRUE(std::fetestexcept(FE_DIVBYZERO));
    }
  }
}

// RTZ must be implemented by the instruction, not by changing host rounding.
TEST(Interp16, P10RtzAcrossHostRoundingModes) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
    ASSERT_EQ(std::fesetround(rounding), 0);
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned m = 0; m < 64; ++m) {
        uint32_t words[4][32], expected[32];
        goc_test::interp16_capture_inputs(words, m * 32);
        for (unsigned lane = 0; lane < 32; ++lane)
          expected[lane] = goc_test::interp16_canonical(
              2, m,
              goc_test::interp16_reference(2, words[0], words[1], words[2], words[3][lane], lane, m,
                                           false));
        const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
        uint32_t *d[] = {words[3]};
        std::feraiseexcept(FE_DIVBYZERO);
        ASSERT_EQ(functions[2](cpu, UINT32_MAX, goc_test::interp16_mode(2, m), d, a, b, c),
                  GOC_SUCCESS);
        EXPECT_EQ(std::fegetround(), rounding);
        EXPECT_TRUE(std::fetestexcept(FE_DIVBYZERO));
        for (unsigned lane = 0; lane < 32; ++lane)
          ASSERT_EQ(goc_test::interp16_canonical(2, m, words[3][lane]), expected[lane])
              << rounding << "/" << cpu << "/" << m << "/" << lane;
      }
  }
}

TEST(Interp16, Wave64HardwareModifiersRoundingAndOverflow) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned op = 0; op < 4; ++op)
      for (unsigned ovfl = 0; ovfl < 2; ++ovfl)
        for (unsigned m = 0; m < 64; ++m) {
          uint64_t digest = goc_test::capture_hash_seed;
          for (unsigned start = 0; start < 1024; start += 64) {
            uint32_t words[4][64];
            wave64_capture_inputs(words, start);
            const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
            uint32_t *d[] = {words[3]};
            ASSERT_EQ(wave64_functions[op](cpu | (ovfl ? GOC_FP16_OVFL : 0), UINT64_MAX,
                                           goc_test::interp16_mode(op, m, 7), d, a, b, c),
                      GOC_SUCCESS);
            for (uint32_t value : words[3]) {
              value = goc_test::interp16_canonical(op, m, value);
              digest = goc_test::capture_hash_bytes(digest, value, 4);
            }
          }
          EXPECT_EQ(digest, goc_test::interp16_digests[(op * 2 + ovfl) * 64 + m])
              << cpu << "/" << op << "/" << ovfl << "/" << m;
        }
}

TEST(Interp16, Wave64MasksAliasesUnalignedStorageAndWaits) {
  const unsigned sources[][3] = {{0, 1, 2}, {0, 0, 2}, {0, 1, 0}, {0, 1, 1}, {0, 0, 0}};
  for (unsigned op = 0; op < 4; ++op)
    for (unsigned m = 0; m < 64; ++m)
      for (unsigned ovfl = 0; ovfl < 2; ++ovfl)
        for (const auto &source : sources)
          for (unsigned target = 0; target < 4; ++target) {
            uint32_t initial[4][67], result[64];
            for (unsigned reg = 0; reg < 4; ++reg)
              for (unsigned lane = 0; lane < 67; ++lane)
                initial[reg][lane] = (lane * 0x7395a831u) ^ (reg * 0xa7925163u);
            for (unsigned lane = 0; lane < 64; ++lane)
              result[lane] = goc_test::interp16_reference(
                  op, initial[source[0]] + 1, initial[source[1]] + 1, initial[source[2]] + 1,
                  initial[target][lane + 1], lane, m, ovfl);
            for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
              for (uint64_t exec_mask : {uint64_t(UINT64_MAX), uint64_t(0xffffffff00000000ULL),
                                         uint64_t(0x87654321fedcba98ULL)}) {
                uint32_t words[4][67];
                std::memcpy(words, initial, sizeof(words));
                const uint32_t *a[] = {words[source[0]] + 1}, *b[] = {words[source[1]] + 1},
                               *c[] = {words[source[2]] + 1};
                uint32_t *d[] = {words[target] + 1};
                ASSERT_EQ(wave64_functions[op](cpu | (ovfl ? GOC_FP16_OVFL : 0), exec_mask,
                                               goc_test::interp16_mode(op, m, m % 8), d, a, b, c),
                          GOC_SUCCESS);
                for (unsigned reg = 0; reg < 4; ++reg)
                  for (unsigned lane = 0; lane < 67; ++lane) {
                    uint32_t want = initial[reg][lane], got = words[reg][lane];
                    if (reg == target && lane > 0 && lane <= 64 &&
                        ((exec_mask >> (lane - 1)) & 1)) {
                      ASSERT_TRUE(compare_result(op, m, got, result[lane - 1]))
                          << cpu << "/" << op << "/" << m << "/" << target << "/" << lane;
                      continue;
                    }
                    ASSERT_EQ(got, want)
                        << cpu << "/" << op << "/" << m << "/" << target << "/" << lane;
                  }
              }
          }
}

TEST(Interp16, Wave64IndependentIntegerOracleAndRoundingBoundaries) {
  const uint32_t edges[] = {0,          0x80000000, 1,          0x3f800000, 0x3f800001, 0x3f800002,
                            0xbf800001, 0x3a000000, 0x33800000, 0x7f7fffff, 0xff7fffff, 0x00800000,
                            0x00013c00, 0x3c010001, 0x7bfffbff, 0x7c007e00};
  for (unsigned op = 0; op < 4; ++op)
    for (unsigned m = 0; m < 64; ++m)
      for (unsigned ovfl = 0; ovfl < 2; ++ovfl) {
        std::mt19937 random(314159);
        for (unsigned batch = 0; batch < 24; ++batch) {
          uint32_t initial[4][64], expected[64];
          for (unsigned reg = 0; reg < 4; ++reg)
            for (unsigned lane = 0; lane < 64; ++lane)
              initial[reg][lane] =
                  batch < 16 ? edges[(batch + reg * 7 + lane) % 16] : uint32_t(random());
          for (unsigned lane = 0; lane < 64; ++lane)
            expected[lane] = goc_test::interp16_canonical(
                op, m,
                goc_test::interp16_reference(op, initial[0], initial[1], initial[2],
                                             initial[3][lane], lane, m, ovfl));
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
            uint32_t words[4][64];
            std::memcpy(words, initial, sizeof(words));
            const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
            uint32_t *d[] = {words[3]};
            ASSERT_EQ(wave64_functions[op](cpu | (ovfl ? GOC_FP16_OVFL : 0), UINT64_MAX,
                                           goc_test::interp16_mode(op, m), d, a, b, c),
                      GOC_SUCCESS);
            ASSERT_EQ(std::memcmp(words, initial, sizeof(initial[0]) * 3), 0);
            for (unsigned lane = 0; lane < 64; ++lane)
              ASSERT_TRUE(compare_result(op, m, words[3][lane], expected[lane]))
                  << cpu << "/" << op << "/" << m << "/" << ovfl << "/" << batch << "/" << lane;
          }
        }
      }
}

TEST(Interp16, Wave64ValidationAndUnchangedHostRounding) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  std::fesetround(FE_TONEAREST);
  for (unsigned op = 0; op < 4; ++op) {
    uint32_t known = goc_test::interp16_mode(op, 63, 7);
    EXPECT_EQ(wave64_functions[op](0, 0U, known, nullptr, nullptr, nullptr, nullptr), GOC_SUCCESS);
    for (unsigned bit = 0; bit < 32; ++bit) {
      if (!(known & (1u << bit))) {
        EXPECT_EQ(wave64_functions[op](0, 0, 1u << bit, nullptr, nullptr, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
      }
    }
    EXPECT_EQ(wave64_functions[op](GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0,
                                   nullptr, nullptr, nullptr, nullptr),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t words[4][64];
      wave64_capture_inputs(words, 0);
      const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
      uint32_t *d[] = {words[3]};
      std::feraiseexcept(FE_DIVBYZERO);
      EXPECT_EQ(wave64_functions[op](cpu, UINT64_MAX, known, d, a, b, c), GOC_SUCCESS);
      EXPECT_EQ(std::fegetround(), FE_TONEAREST);
      EXPECT_TRUE(std::fetestexcept(FE_DIVBYZERO));
    }
  }
}

// RTZ must be implemented by the instruction, not by changing host rounding.
TEST(Interp16, Wave64P10RtzAcrossHostRoundingModes) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
    ASSERT_EQ(std::fesetround(rounding), 0);
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned m = 0; m < 64; ++m) {
        uint32_t words[4][64], expected[64];
        wave64_capture_inputs(words, m * 32);
        for (unsigned lane = 0; lane < 64; ++lane)
          expected[lane] = goc_test::interp16_canonical(
              2, m,
              goc_test::interp16_reference(2, words[0], words[1], words[2], words[3][lane], lane, m,
                                           false));
        const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
        uint32_t *d[] = {words[3]};
        std::feraiseexcept(FE_DIVBYZERO);
        ASSERT_EQ(wave64_functions[2](cpu, UINT64_MAX, goc_test::interp16_mode(2, m), d, a, b, c),
                  GOC_SUCCESS);
        EXPECT_EQ(std::fegetround(), rounding);
        EXPECT_TRUE(std::fetestexcept(FE_DIVBYZERO));
        for (unsigned lane = 0; lane < 64; ++lane)
          ASSERT_EQ(goc_test::interp16_canonical(2, m, words[3][lane]), expected[lane])
              << rounding << "/" << cpu << "/" << m << "/" << lane;
      }
  }
}

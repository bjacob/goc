// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "fp_environment.h"
#include "fuzzy.h"
#include "goc/goc.h"
#include "half_reference.h"
#include "loose_tolerances.h"
#include "scalar_fp_hardware.h"
#include "scalar_fp_reference.h"

#include <cfenv>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdint.h>

namespace {

float value(uint32_t bits) {
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

uint32_t arithmetic_reference(unsigned op, uint32_t a, uint32_t b, uint64_t flags) {
  bool half = op & 1;
  double x = half ? goc_test::half_value(uint16_t(a)) : double(value(a));
  double y = half ? goc_test::half_value(uint16_t(b)) : double(value(b));
  double minimum = half ? 0x1p-14 : 0x1p-126;
  if (flags & GOC_FP_FLUSH_INPUT_DENORMALS) {
    if (std::abs(x) < minimum)
      x = std::copysign(0.0, x);
    if (std::abs(y) < minimum)
      y = std::copysign(0.0, y);
  }
  double result = op < 2 ? x + y : op < 4 ? x - y : x * y;
  // Guest tininess is tested before rounding, at the midpoint below min-normal.
  if ((flags & GOC_FP_FLUSH_OUTPUT_DENORMALS) &&
      std::abs(result) < minimum - (half ? 0x1p-26 : 0x1p-151))
    result = std::copysign(0.0, result);
  if (half)
    return goc_test::half_bits(result, flags & GOC_FP16_OVFL);
  float rounded = float(result);
  uint32_t bits;
  std::memcpy(&bits, &rounded, sizeof(bits));
  return bits;
}

::testing::AssertionResult close(uint32_t actual, uint32_t expected, bool half) {
  unsigned magnitude = expected & (half ? 0x7fffU : 0x7fffffffU);
  if ((half && (actual >> 16)) || !magnitude || magnitude == (half ? 0x7c00U : 0x7f800000U)) {
    if (actual == expected)
      return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure()
           << "actual bits=" << actual << ", expected bits=" << expected;
  }
  return half ? goc_test::fuzzy(goc_test::half_value(uint16_t(actual)),
                                goc_test::half_value(uint16_t(expected)),
                                goc_test::scalar_arithmetic_f16_tolerance)
              : goc_test::fuzzy(value(actual), value(expected),
                                goc_test::scalar_arithmetic_f32_tolerance);
}

} // namespace

TEST(ScalarFp, HardwareEveryFpStateAndUnderflowBoundary) {
  for (unsigned dataset = 0; dataset < 2; ++dataset)
    for (unsigned state = 0; state < 8; ++state)
      for (unsigned op = 0; op < 14; ++op)
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
          uint64_t hash = goc_test::capture_hash_seed;
          for (unsigned i = 0; i < 4096; ++i) {
            uint32_t w[2], d;
            if (dataset)
              goc_test::scalar_fp_boundary_inputs(i, op & 1, w);
            else
              goc_test::scalar_fp_inputs(i, op & 1, w);
            ASSERT_EQ(goc_test::scalar_fp_functions[op](cpu | goc_test::scalar_fp_flags(state), 0,
                                                        &d, w[0], w[1], nullptr),
                      GOC_SUCCESS);
            uint32_t reference = d;
            if (op < 6) {
              reference = arithmetic_reference(op, w[0], w[1], goc_test::scalar_fp_flags(state));
              ASSERT_TRUE(close(d, reference, op & 1))
                  << dataset << "/" << state << "/" << op << "/" << cpu << "/" << i;
            }
            // Validate the computed arithmetic oracle against the independent
            // capture. Selection operations still require raw output equality.
            hash =
                goc_test::capture_hash_word(hash, goc_test::scalar_fp_canonical(reference, op & 1));
          }
          ASSERT_EQ(hash, goc_test::scalar_fp_hardware[dataset][state][op])
              << dataset << "/" << state << "/" << op << "/" << cpu;
        }
}

TEST(ScalarFp, InputStorageMayAlias) {
  for (unsigned state = 0; state < 8; ++state)
    for (unsigned op = 0; op < 14; ++op)
      for (unsigned sample : {0u, 3u, 216u, 774u, 1023u, 3001u}) {
        uint32_t w[2], expected;
        goc_test::scalar_fp_inputs(sample, op & 1, w);
        auto flags = goc_test::scalar_fp_flags(state);
        ASSERT_EQ(goc_test::scalar_fp_functions[op](flags, 0, &expected, w[0], w[1], nullptr),
                  GOC_SUCCESS);
        {
          uint32_t storage[] = {123, w[0], w[1], 456};
          ASSERT_EQ(goc_test::scalar_fp_functions[op](flags, 0, storage + 1, storage[1], storage[2],
                                                      nullptr),
                    GOC_SUCCESS);
          EXPECT_EQ(goc_test::scalar_fp_canonical(storage[1], op & 1),
                    goc_test::scalar_fp_canonical(expected, op & 1));
          EXPECT_EQ(storage[0], 123u);
          EXPECT_EQ(storage[2], w[1]);
          EXPECT_EQ(storage[3], 456u);
        }
      }
}

TEST(ScalarFp, DenormalStagesAndFiniteOverflow) {
  uint32_t d;
  const uint64_t output = GOC_FP_FLUSH_OUTPUT_DENORMALS;
  ASSERT_EQ(goc_s_mul_f32(output, 0, &d, 0x00800000, 0x3f7fffff, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0u);
  ASSERT_EQ(goc_s_mul_f32(0, 0, &d, 0x00800000, 0x3f7fffff, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0x00800000u);
  ASSERT_EQ(goc_s_mul_f32(output, 0, &d, 0x007fffff, 0x3f800001, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0x00800000u);
  ASSERT_EQ(goc_s_mul_f16(output, 0, &d, 0x0400, 0x3bff, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0u);
  ASSERT_EQ(goc_s_mul_f16(output, 0, &d, 0x03ff, 0x3c01, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0x0400u);
  ASSERT_EQ(goc_s_min_num_f32(output, 0, &d, 0x80000001, 0, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0x80000001u);
  ASSERT_EQ(goc_s_min_num_f32(GOC_FP_FLUSH_INPUT_DENORMALS, 0, &d, 0x80000001, 0, nullptr),
            GOC_SUCCESS);
  EXPECT_EQ(d, 0x80000000u);
  ASSERT_EQ(goc_s_mul_f16(GOC_FP16_OVFL, 0, &d, 0x7bff, 0x4000, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0x7bffu);
  ASSERT_EQ(goc_s_mul_f16(GOC_FP16_OVFL, 0, &d, 0x7c00, 0x4000, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0x7c00u);
}

TEST(ScalarFp, ErrorsLeaveOutputUntouchedAndRoundingPreserved) {
  goc_test::ScopedFpEnvironment restore;
  ASSERT_TRUE(restore.saved());
  ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
  for (unsigned op = 0; op < 14; ++op) {
    uint32_t d = 123;
    for (unsigned bit = 0; bit < 32; ++bit)
      EXPECT_EQ(goc_test::scalar_fp_functions[op](0, 1u << bit, &d, 0, 0, nullptr),
                GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_test::scalar_fp_functions[op](1ULL << 63, 0, &d, 0, 0, nullptr),
              GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_test::scalar_fp_functions[op](
                  GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, &d, 0, 0, nullptr),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    EXPECT_EQ(d, 123u);
    EXPECT_EQ(goc_test::scalar_fp_functions[op](0, 0, &d, 0, 0, nullptr), GOC_SUCCESS);
    EXPECT_EQ(std::fegetround(), FE_TONEAREST);
  }
}

TEST(ScalarFp, MinmaxLiteralZerosNaNsAndSubnormals) {
  for (unsigned half = 0; half < 2; ++half) {
    const uint32_t sign = half ? 0x8000 : 0x80000000;
    const uint32_t one = half ? 0x3c00 : 0x3f800000;
    const uint32_t two = half ? 0x4000 : 0x40000000;
    const uint32_t inf = half ? 0x7c00 : 0x7f800000;
    const uint32_t quiet = half ? 0x200 : 0x400000;
    // A, B, numeric minimum, numeric maximum. NaN cases are checked separately.
    const uint32_t samples[][4] = {{sign, 0, sign, 0},
                                   {0, sign, sign, 0},
                                   {one, two, one, two},
                                   {one | sign, two | sign, two | sign, one | sign},
                                   {inf, inf | sign, inf | sign, inf},
                                   {inf | quiet | 1, one, one, one},
                                   {one, inf | 1, one, one},
                                   {inf | 1, inf | quiet | 2, 0, 0},
                                   {1, 0, 0, 1},
                                   {sign | 1, 0, sign | 1, 0}};
    for (unsigned operation = 0; operation < 4; ++operation)
      for (unsigned index = 0; index < 10; ++index)
        for (unsigned state = 0; state < 8; ++state)
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
            uint64_t flags = cpu | goc_test::scalar_fp_flags(state);
            uint32_t a = samples[index][0], b = samples[index][1];
            uint32_t want = samples[index][2 + (operation & 1)];
            bool nan = index == 7 || (operation >= 2 && (index == 5 || index == 6));
            if (index >= 8 && (flags & GOC_FP_FLUSH_INPUT_DENORMALS))
              want &= sign;
            for (unsigned alias : {1U, 2U}) {
              uint32_t words[] = {123, a | (half ? 0xfc010000U : 0), b | (half ? 0x7c010000U : 0),
                                  456};
              uint32_t state_out = 0xa5000040;
              ASSERT_EQ(goc_test::scalar_fp_functions[6 + 2 * operation + half](
                            flags, 0, words + alias, words[1], words[2], &state_out),
                        GOC_SUCCESS);
              if (nan)
                EXPECT_GT(words[alias] & (sign - 1), inf);
              else
                EXPECT_EQ(words[alias], want) << operation << "/" << index << "/" << state;
              if (half) {
                EXPECT_EQ(words[alias] >> 16, 0U);
              }
              EXPECT_EQ(words[0], 123U);
              EXPECT_EQ(words[3], 456U);
              EXPECT_EQ(words[3 - alias],
                        (alias == 1 ? b : a) |
                            (half ? (alias == 1 ? 0x7c010000U : 0xfc010000U) : 0));
              EXPECT_EQ(state_out, 0xa5000040U);
            }
          }
  }
}

TEST(ScalarFp, LooseFallbackAndReportingErrorPrecedence) {
  for (auto fn : goc_test::scalar_fp_functions) {
    for (unsigned state = 0; state < 8; ++state) {
      uint64_t flags = goc_test::scalar_fp_flags(state);
      uint32_t result, fallback, exceptions = 0xa5000040;
      ASSERT_EQ(fn(flags, 0, &result, 0x3fc03e00, 0xbf80bc00, &exceptions), GOC_SUCCESS);
      EXPECT_EQ(exceptions, 0xa5000040U);
      ASSERT_EQ(
          fn(flags | GOC_SEMANTICS_EXACT_EMPIRICAL, 0, &fallback, 0x3fc03e00, 0xbf80bc00, nullptr),
          GOC_SUCCESS);
      EXPECT_EQ(fallback, result);
    }
    for (uint64_t semantics : {GOC_SEMANTICS_EXACT_EMPIRICAL, 2 * GOC_SEMANTICS_EXACT_EMPIRICAL,
                               3 * GOC_SEMANTICS_EXACT_EMPIRICAL})
      for (uint64_t strict : {uint64_t(0), GOC_SEMANTICS_STRICT})
        for (uint64_t mode : {uint64_t(0), uint64_t(GOC_ALU_NEG_A), uint64_t(1) << 63}) {
          uint32_t output = 0xdeadbeef, exceptions = 0xa5000040;
          EXPECT_EQ(fn(semantics | strict, mode, &output, 0, 0, &exceptions),
                    GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
          EXPECT_EQ(output, 0xdeadbeefU);
          EXPECT_EQ(exceptions, 0xa5000040U);
        }
    for (unsigned bit = 0; bit < 64; ++bit) {
      uint32_t output = 0xdeadbeef, exceptions = 0xa5000040;
      EXPECT_EQ(fn(0, 1ULL << bit, &output, 0, 0, &exceptions), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(output, 0xdeadbeefU);
      EXPECT_EQ(exceptions, 0xa5000040U);
    }
  }
}

// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "fp_environment.h"
#include "fuzzy.h"
#include "goc/goc.h"
#include "half_fma_reference.h"
#include "half_reference.h"
#include "loose_tolerances.h"
#include "scalar_fma_hardware.h"
#include "scalar_fma_reference.h"
#include "scalar_fp_reference.h"

#include <cfenv>
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

::testing::AssertionResult close(uint32_t actual, uint32_t expected, bool half) {
  uint32_t magnitude = expected & (half ? 0x7fffU : 0x7fffffffU);
  uint32_t inf = half ? 0x7c00 : 0x7f800000;
  if (half && (actual >> 16))
    return ::testing::AssertionFailure() << "upper half is not zero";
  if (!magnitude || magnitude == inf) {
    if (actual == expected)
      return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure()
           << "actual bits=" << actual << ", expected bits=" << expected;
  }
  return half ? goc_test::fuzzy(goc_test::half_value(uint16_t(actual)),
                                goc_test::half_value(uint16_t(expected)),
                                goc_test::fma_f16_tolerance)
              : goc_test::fuzzy(value(actual), value(expected), goc_test::fma_f32_tolerance);
}

} // namespace

TEST(ScalarFma, HardwareEveryFpStateAndLiteral) {
  for (unsigned state = 0; state < 8; ++state)
    for (unsigned k = 0; k < 26; ++k)
      for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
          if (k == 1 && semantics)
            continue;
          uint64_t hash = goc_test::capture_hash_seed;
          for (unsigned i = 0; i < 4096; ++i) {
            uint32_t w[3];
            goc_test::scalar_fma_inputs(i, k == 1, w);
            uint32_t d = w[2];
            ASSERT_EQ(goc_test::scalar_fma_call(k,
                                                cpu | semantics | GOC_SEMANTICS_STRICT |
                                                    goc_test::scalar_fp_flags(state),
                                                0, &d, w[0], w[1]),
                      GOC_SUCCESS);
            uint32_t reference = d;
            if (!semantics) {
              if (k == 1) {
                reference = goc_test::half_fma_reference::evaluate(
                    w[0], w[1], w[2], 0, goc_test::scalar_fp_flags(state) & GOC_FP16_OVFL,
                    goc_test::scalar_fp_flags(state));
              } else {
                reference = w[2];
                ASSERT_EQ(goc_test::scalar_fma_call(k,
                                                    GOC_SEMANTICS_EXACT_EMPIRICAL |
                                                        GOC_SEMANTICS_STRICT |
                                                        goc_test::scalar_fp_flags(state),
                                                    0, &reference, w[0], w[1]),
                          GOC_SUCCESS);
              }
              ASSERT_TRUE(close(d, reference, k == 1))
                  << state << "/" << k << "/" << cpu << "/" << i;
            }
            // Keep the independent capture as evidence for the reference used
            // above. Loose outputs need proximity, not an identical digest.
            hash =
                goc_test::capture_hash_word(hash, goc_test::scalar_fp_canonical(reference, k == 1));
          }
          ASSERT_EQ(hash, goc_test::scalar_fma_hardware[state][k])
              << state << "/" << k << "/" << cpu;
        }
}

TEST(ScalarFma, AllScalarAliases) {
  for (unsigned state = 0; state < 8; ++state)
    for (unsigned k = 0; k < 26; ++k)
      for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
        for (unsigned sample : {0u, 6u, 18u, 1023u, 3001u}) {
          if (k == 1 && semantics)
            continue;
          uint32_t w[3];
          goc_test::scalar_fma_inputs(sample, k == 1, w);
          auto flags = goc_test::scalar_fp_flags(state) | semantics | GOC_SEMANTICS_STRICT;
          for (unsigned alias = 0; alias < 3; ++alias) {
            uint32_t expected = w[alias];
            ASSERT_EQ(goc_test::scalar_fma_call(k, flags, 0, &expected, w[0], w[1]), GOC_SUCCESS);
            {
              uint32_t words[] = {123, w[0], w[1], w[2], 456};
              ASSERT_EQ(
                  goc_test::scalar_fma_call(k, flags, 0, words + 1 + alias, words[1], words[2]),
                  GOC_SUCCESS);
              EXPECT_EQ(goc_test::scalar_fp_canonical(words[1 + alias], k == 1),
                        goc_test::scalar_fp_canonical(expected, k == 1));
              for (unsigned j = 0; j < 3; ++j)
                if (j != alias) {
                  EXPECT_EQ(words[j + 1], w[j]);
                }
              EXPECT_EQ(words[0], 123u);
              EXPECT_EQ(words[4], 456u);
            }
          }
        }
}

TEST(ScalarFma, TrueFusionAndGuestFpStages) {
  uint32_t d = 0xbf800000;
  ASSERT_EQ(goc_s_fmac_f32(0, 0, &d, 0x3f800001, 0x3f7ffffe, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0xa8800000u);
  d = 0xabcdbc00;
  ASSERT_EQ(goc_s_fmac_f16(0, 0, &d, 0x3c01, 0x3bfe, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0x8010u);
  d = 0;
  ASSERT_EQ(goc_s_fmac_f32(GOC_FP_FLUSH_OUTPUT_DENORMALS, 0, &d, 0x00800000, 0x3f7fffff, nullptr),
            GOC_SUCCESS);
  EXPECT_EQ(d, 0u);
  d = 0;
  ASSERT_EQ(goc_s_fmac_f32(0, 0, &d, 0x00800000, 0x3f7fffff, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0x00800000u);
  d = 1;
  ASSERT_EQ(goc_s_fmac_f16(GOC_FP_FLUSH_INPUT_DENORMALS, 0, &d, 0x3c00, 0, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0u);
  d = 0;
  ASSERT_EQ(goc_s_fmac_f16(GOC_FP16_OVFL, 0, &d, 0x7bff, 0x4000, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0x7bffu);
  ASSERT_EQ(goc_s_fmaak_f32(0, 0, &d, 0x3f800001, 0x3f7ffffe, 0xbf800000, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0xa8800000u);
  ASSERT_EQ(goc_s_fmamk_f32(0, 0, &d, 0x3f800001, 0x3f7ffffe, 0xbf800000, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0xa8800000u);
}

TEST(ScalarFma, ErrorsAndHostRounding) {
  goc_test::ScopedFpEnvironment restore;
  ASSERT_TRUE(restore.saved());
  ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
  for (unsigned k = 0; k < 26; ++k) {
    uint32_t d = 123;
    for (unsigned bit = 0; bit < 32; ++bit)
      EXPECT_EQ(goc_test::scalar_fma_call(k, 0, 1u << bit, &d, 0, 0), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_test::scalar_fma_call(k, 1ULL << 63, 0, &d, 0, 0), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(d, 123U);
    if (k == 1) {
      EXPECT_EQ(goc_test::scalar_fma_call(k, GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                                          0, &d, 0, 0),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
      EXPECT_EQ(d, 123u);
    }
    EXPECT_EQ(goc_test::scalar_fma_call(k, 0, 0, &d, 0, 0), GOC_SUCCESS);
    EXPECT_EQ(std::fegetround(), FE_TONEAREST);
  }
}

TEST(ScalarFma, ExactFp32LiteralNaNsFusionAndTininess) {
  // A, B, C, preserved output, flushed output. Source inputs here are normal,
  // zero or special, so only output flushing changes these expected results.
  const uint32_t cases[][5] = {{0x3f800001, 0x3f7ffffe, 0xbf800000, 0xa8800000, 0xa8800000},
                               {0, 0x7f800000, 0xffc00123, 0xffc00000, 0xffc00000},
                               {0x7fc00123, 0xff800456, 0x3f800000, 0x7fc00123, 0x7fc00123},
                               {0x3f800000, 0xff800456, 0x7fc00789, 0xffc00456, 0xffc00456},
                               {0x3f800000, 0x3f800000, 0xff800789, 0xffc00789, 0xffc00789},
                               {0x7f800000, 0x3f800000, 0xff800000, 0xffc00000, 0xffc00000},
                               {0x80000000, 0x3f800000, 0x80000000, 0x80000000, 0x80000000},
                               {0, 0x3f800000, 0x80000000, 0, 0},
                               {0x00800000, 0x3f7fffff, 0, 0x00800000, 0}};
  for (unsigned op = 0; op < 3; ++op)
    for (unsigned state = 0; state < 8; ++state)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (const auto &sample : cases) {
          uint64_t flags = cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT |
                           goc_test::scalar_fp_flags(state);
          uint32_t words[] = {123, sample[2], 456};
          int status =
              op == 0 ? goc_s_fmac_f32(flags, 0, words + 1, sample[0], sample[1], nullptr)
              : op == 1
                  ? goc_s_fmaak_f32(flags, 0, words + 1, sample[0], sample[1], sample[2], nullptr)
                  : goc_s_fmamk_f32(flags, 0, words + 1, sample[0], sample[1], sample[2], nullptr);
          ASSERT_EQ(status, GOC_SUCCESS);
          EXPECT_EQ(words[1], sample[3 + bool(flags & GOC_FP_FLUSH_OUTPUT_DENORMALS)]);
          EXPECT_EQ(words[0], 123U);
          EXPECT_EQ(words[2], 456U);
        }
}

TEST(ScalarFma, ReportingErrorsPrecedeAccumulatorReads) {
  for (unsigned op = 0; op < 4; ++op) {
    auto call = [&](uint64_t flags, uint64_t mode, uint32_t *d, uint32_t *state) {
      if (op == 0)
        return goc_s_fmac_f32(flags, mode, d, 0, 0, state);
      if (op == 1)
        return goc_s_fmac_f16(flags, mode, d, 0, 0, state);
      if (op == 2)
        return goc_s_fmaak_f32(flags, mode, d, 0, 0, 0, state);
      return goc_s_fmamk_f32(flags, mode, d, 0, 0, 0, state);
    };
    for (uint64_t semantics : {GOC_SEMANTICS_EXACT_EMPIRICAL, 2 * GOC_SEMANTICS_EXACT_EMPIRICAL,
                               3 * GOC_SEMANTICS_EXACT_EMPIRICAL})
      for (uint64_t strict : {uint64_t(0), GOC_SEMANTICS_STRICT})
        for (uint64_t mode : {uint64_t(0), uint64_t(GOC_ALU_NEG_A), uint64_t(1) << 63}) {
          uint32_t d = 0xdeadbeef, state = 0xa5000040;
          EXPECT_EQ(call(semantics | strict, mode, &d, &state), GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
          EXPECT_EQ(call(semantics | strict, mode, nullptr, &state),
                    GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
          EXPECT_EQ(d, 0xdeadbeefU);
          EXPECT_EQ(state, 0xa5000040U);
        }
    uint32_t d = 0, state = 0xa5000040;
    EXPECT_EQ(call(0, 0, &d, &state), GOC_SUCCESS);
    EXPECT_EQ(state, 0xa5000040U);
  }
}

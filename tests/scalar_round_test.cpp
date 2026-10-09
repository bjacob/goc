// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "scalar_fp_reference.h"
#include "scalar_round_exceptions_hardware.h"
#include "scalar_round_hardware.h"
#include "scalar_round_reference.h"

#include <cfenv>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdint.h>

TEST(ScalarRound, HardwareAllHalfPatternsAndFpStates) {
  for (unsigned state = 0; state < 8; ++state)
    for (unsigned op = 0; op < 8; ++op)
      for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
          uint64_t hash = goc_test::capture_hash_seed;
          for (unsigned i = 0; i < 65536; ++i) {
            uint32_t w[2], d;
            goc_test::scalar_round_inputs(i, op & 1, w);
            ASSERT_EQ(goc_test::scalar_round_functions[op](cpu | semantics | GOC_SEMANTICS_STRICT |
                                                               goc_test::scalar_fp_flags(state),
                                                           0, &d, w[0], nullptr),
                      GOC_SUCCESS);
            hash = goc_test::capture_hash_word(hash, d);
          }
          ASSERT_EQ(hash, goc_test::scalar_round_hardware[state][op])
              << state << "/" << op << "/" << semantics << "/" << cpu;
        }
}

TEST(ScalarRound, Aliasing) {
  for (unsigned state = 0; state < 8; ++state)
    for (unsigned op = 0; op < 8; ++op)
      for (unsigned i : {0u, 1u, 8u, 10u, 12u, 30u, 0x7c01u, 0x7e01u, 0x8001u, 0xffffu}) {
        uint32_t w[2], expected;
        goc_test::scalar_round_inputs(i, op & 1, w);
        uint64_t flags = goc_test::scalar_fp_flags(state);
        ASSERT_EQ(goc_test::scalar_round_functions[op](flags, 0, &expected, w[0], nullptr),
                  GOC_SUCCESS);
        {
          uint32_t words[] = {123, w[0], 456};
          ASSERT_EQ(goc_test::scalar_round_functions[op](flags, 0, words + 1, words[1], nullptr),
                    GOC_SUCCESS);
          EXPECT_EQ(words[1], expected);
          EXPECT_EQ(words[0], 123u);
          EXPECT_EQ(words[2], 456u);
        }
      }
}

TEST(ScalarRound, HostFpStatePreservedForEveryRoundingMode) {
  goc_test::ScopedFpEnvironment restore;
  ASSERT_TRUE(restore.saved());
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    ASSERT_EQ(std::fesetround(rounding), 0);
    std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_DIVBYZERO);
    int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
    for (unsigned op = 0; op < 8; ++op)
      for (uint32_t a : {0u, 1u, 0x80000001u, 0x3fc00000u, 0x7f800001u, 0x7c01u, 0x3e00u}) {
        uint32_t d;
        EXPECT_EQ(
            goc_test::scalar_round_functions[op](GOC_SEMANTICS_EXACT_EMPIRICAL, 0, &d, a, nullptr),
            GOC_SUCCESS);
        EXPECT_EQ(std::fegetround(), rounding);
        EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
      }
    uint32_t d;
    EXPECT_EQ(goc_s_rndne_f32(0, 0, &d, 0x3fc00000, nullptr), GOC_SUCCESS);
    EXPECT_EQ(d, 0x40000000u);
    EXPECT_EQ(goc_s_rndne_f16(0, 0, &d, 0x4100, nullptr), GOC_SUCCESS);
    EXPECT_EQ(d, 0x4000u);
  }
}

TEST(ScalarRound, NaNsZerosAndInputFlushing) {
  uint32_t d;
  ASSERT_EQ(goc_s_ceil_f32(0, 0, &d, 0xff800123, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0xffc00123u);
  ASSERT_EQ(goc_s_floor_f16(0, 0, &d, 0xabcdfc01, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0xfe01u);
  ASSERT_EQ(goc_s_ceil_f32(0, 0, &d, 0x80000001, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0x80000000u);
  ASSERT_EQ(goc_s_floor_f16(0, 0, &d, 0x8001, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0xbc00u);
  ASSERT_EQ(goc_s_floor_f16(GOC_FP_FLUSH_INPUT_DENORMALS, 0, &d, 0x8001, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0x8000u);
  ASSERT_EQ(goc_s_ceil_f32(GOC_FP_FLUSH_OUTPUT_DENORMALS, 0, &d, 1, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0x3f800000u);
}

TEST(ScalarRound, ErrorsDoNotWrite) {
  for (unsigned op = 0; op < 8; ++op) {
    uint32_t d = 123;
    for (unsigned bit = 0; bit < 32; ++bit)
      EXPECT_EQ(goc_test::scalar_round_functions[op](0, 1u << bit, &d, 0, nullptr),
                GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_test::scalar_round_functions[op](1ULL << 63, 0, &d, 0, nullptr),
              GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(d, 123u);
  }
}

TEST(ScalarRound, HardwareExceptionFlags) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned denorm = 0; denorm < 4; ++denorm)
      for (unsigned op = 0; op < 8; ++op) {
        uint64_t hash = goc_test::capture_hash_seed;
        for (unsigned i = 0; i < 65536; ++i) {
          uint32_t a =
              op & 1 ? (0xabcd0000U | i) : (i << 16) | ((i * 1664525U + 1013904223U) & 65535);
          uint64_t flags = cpu | GOC_SEMANTICS_EXACT_EMPIRICAL |
                           (i & 2 ? GOC_SEMANTICS_STRICT : 0) |
                           (denorm & 1 ? 0 : GOC_FP_FLUSH_INPUT_DENORMALS) |
                           (denorm & 2 ? 0 : GOC_FP_FLUSH_OUTPUT_DENORMALS);
          uint32_t d = 0, excp_flag_user = 0xa5000040;
          ASSERT_EQ(goc_test::scalar_round_functions[op](flags, 0, &d, a, &excp_flag_user),
                    GOC_SUCCESS);
          ASSERT_EQ(excp_flag_user & ~3U, 0xa5000040U);
          hash = goc_test::capture_hash_word(hash, excp_flag_user & 3);
        }
        EXPECT_EQ(hash, goc_test::scalar_round_exception_hashes[denorm & 1][op & 1 ? 0 : 1]) << op;
      }
}

TEST(ScalarRound, ReportingContractAndHostState) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (unsigned op = 0; op < 8; ++op) {
    uint32_t a = op & 1 ? 0x7c01 : 0x7f800001;
    for (int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
      std::fesetround(rounding);
      std::feraiseexcept(FE_INEXACT);
      int host_exceptions = std::fetestexcept(FE_ALL_EXCEPT);
      uint32_t d = 0xdeadbeef, excp_flag_user = 0x80000040;
      auto fn = goc_test::scalar_round_functions[op];
      EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL, 1, &d, a, &excp_flag_user),
                GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(d, 0xdeadbeefU);
      EXPECT_EQ(excp_flag_user, 0x80000040U);
      EXPECT_EQ(fn(GOC_SEMANTICS_LOOSE, 0, &d, a, &excp_flag_user), GOC_SUCCESS);
      EXPECT_EQ(excp_flag_user, 0x80000040U);
      EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL, 0, &d, a, &excp_flag_user), GOC_SUCCESS);
      EXPECT_EQ(excp_flag_user, 0x80000040U | GOC_EXCEPTION_INVALID);
      EXPECT_EQ(std::fegetround(), rounding);
      EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), host_exceptions);
    }
  }
}

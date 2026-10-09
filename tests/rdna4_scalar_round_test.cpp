// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "rdna4_exec_masks.h"
#include "rdna4_scalar_fp_reference.h"
#include "rdna4_scalar_round_hardware.h"
#include "rdna4_scalar_round_reference.h"

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
                                                           UINT32_MAX, 0, &d, w[0]),
                      GOC_SUCCESS);
            hash = goc_test::capture_hash_word(hash, d);
          }
          ASSERT_EQ(hash, goc_test::scalar_round_hardware[state][op])
              << state << "/" << op << "/" << semantics << "/" << cpu;
        }
}

TEST(ScalarRound, ExecAndAliasing) {
  for (unsigned state = 0; state < 8; ++state)
    for (unsigned op = 0; op < 8; ++op)
      for (unsigned i : {0u, 1u, 8u, 10u, 12u, 30u, 0x7c01u, 0x7e01u, 0x8001u, 0xffffu}) {
        uint32_t w[2], expected;
        goc_test::scalar_round_inputs(i, op & 1, w);
        uint64_t flags = goc_test::scalar_fp_flags(state);
        ASSERT_EQ(goc_test::scalar_round_functions[op](flags, UINT32_MAX, 0, &expected, w[0]),
                  GOC_SUCCESS);
        for (uint32_t mask : rdna4_exec_masks()) {
          uint32_t words[] = {123, w[0], 456};
          ASSERT_EQ(goc_test::scalar_round_functions[op](flags, mask, 0, words + 1, words[1]),
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
        EXPECT_EQ(goc_test::scalar_round_functions[op](GOC_SEMANTICS_EXACT_EMPIRICAL, 0, 0, &d, a),
                  GOC_SUCCESS);
        EXPECT_EQ(std::fegetround(), rounding);
        EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
      }
    uint32_t d;
    EXPECT_EQ(goc_rdna4_s_rndne_f32(0, 0, 0, &d, 0x3fc00000), GOC_SUCCESS);
    EXPECT_EQ(d, 0x40000000u);
    EXPECT_EQ(goc_rdna4_s_rndne_f16(0, 0, 0, &d, 0x4100), GOC_SUCCESS);
    EXPECT_EQ(d, 0x4000u);
  }
}

TEST(ScalarRound, NaNsZerosAndInputFlushing) {
  uint32_t d;
  ASSERT_EQ(goc_rdna4_s_ceil_f32(0, 0, 0, &d, 0xff800123), GOC_SUCCESS);
  EXPECT_EQ(d, 0xffc00123u);
  ASSERT_EQ(goc_rdna4_s_floor_f16(0, 0, 0, &d, 0xabcdfc01), GOC_SUCCESS);
  EXPECT_EQ(d, 0xfe01u);
  ASSERT_EQ(goc_rdna4_s_ceil_f32(0, 0, 0, &d, 0x80000001), GOC_SUCCESS);
  EXPECT_EQ(d, 0x80000000u);
  ASSERT_EQ(goc_rdna4_s_floor_f16(0, 0, 0, &d, 0x8001), GOC_SUCCESS);
  EXPECT_EQ(d, 0xbc00u);
  ASSERT_EQ(goc_rdna4_s_floor_f16(GOC_FP_FLUSH_INPUT_DENORMALS, 0, 0, &d, 0x8001), GOC_SUCCESS);
  EXPECT_EQ(d, 0x8000u);
  ASSERT_EQ(goc_rdna4_s_ceil_f32(GOC_FP_FLUSH_OUTPUT_DENORMALS, 0, 0, &d, 1), GOC_SUCCESS);
  EXPECT_EQ(d, 0x3f800000u);
}

TEST(ScalarRound, ErrorsDoNotWrite) {
  for (unsigned op = 0; op < 8; ++op) {
    uint32_t d = 123;
    for (unsigned bit = 0; bit < 32; ++bit)
      EXPECT_EQ(goc_test::scalar_round_functions[op](0, 0, 1u << bit, &d, 0),
                GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_test::scalar_round_functions[op](1ULL << 63, 0, 0, &d, 0),
              GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(d, 123u);
  }
}

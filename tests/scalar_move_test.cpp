// SPDX-License-Identifier: MIT

#include "fp_environment.h"
#include "goc/goc.h"

#include <cfenv>
#include <gtest/gtest.h>
#include <stdint.h>

TEST(ScalarMove, RawBitsConditionsAndAliases) {
  goc_test::ScopedFpEnvironment restore;
  ASSERT_TRUE(restore.saved());
  ASSERT_EQ(std::fesetround(FE_UPWARD), 0);
  ASSERT_EQ(std::feraiseexcept(FE_INVALID), 0);
  int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
  for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
    for (unsigned bit = 0; bit < 64; ++bit)
      for (uint32_t scc : {0U, 1U, 0xfffffffeU, 0xffffffffU}) {
        uint64_t a = 1ULL << bit, d64 = ~a;
        uint32_t a32 = uint32_t(a), d32 = ~a32;
        uint64_t flags = semantics | GOC_SEMANTICS_STRICT;
        ASSERT_EQ(goc_s_cmov_b64(flags, 0, &d64, a, scc), GOC_SUCCESS);
        EXPECT_EQ(d64, (scc & 1) ? a : ~a);
        ASSERT_EQ(goc_s_cmov_b32(flags, 0, &d32, a32, scc), GOC_SUCCESS);
        EXPECT_EQ(d32, (scc & 1) ? a32 : ~a32);
        ASSERT_EQ(goc_s_mov_b64(flags, 0, &d64, a), GOC_SUCCESS);
        ASSERT_EQ(goc_s_mov_b32(flags, 0, &d32, a32), GOC_SUCCESS);
        EXPECT_EQ(d64, a);
        EXPECT_EQ(d32, a32);
        ASSERT_EQ(goc_s_cmov_b64(flags, 0, &d64, d64, scc), GOC_SUCCESS);
        ASSERT_EQ(goc_s_cmov_b32(flags, 0, &d32, d32, scc), GOC_SUCCESS);
        EXPECT_EQ(d64, a);
        EXPECT_EQ(d32, a32);
      }
  EXPECT_EQ(std::fegetround(), FE_UPWARD);
  EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
}

TEST(ScalarMove, EveryImmediate) {
  for (uint32_t imm = 0; imm < 65536; ++imm)
    for (uint32_t scc : {0U, 1U}) {
      uint32_t words[] = {0xaabbccdd, 0xdeadbeef, 0x12345678};
      uint32_t value = uint32_t(imm < 32768 ? int64_t(imm) : int64_t(imm) - 65536);
      ASSERT_EQ(goc_s_cmovk_i32(GOC_SEMANTICS_EXACT_EMPIRICAL, 0, words + 1, imm, scc),
                GOC_SUCCESS);
      ASSERT_EQ(words[1], scc ? value : 0xdeadbeefU);
      ASSERT_EQ(goc_s_movk_i32(GOC_SEMANTICS_EXACT_EMPIRICAL, 0, words + 1, imm), GOC_SUCCESS);
      ASSERT_EQ(words[1], value);
      ASSERT_EQ(words[0], 0xaabbccddU);
      ASSERT_EQ(words[2], 0x12345678U);
    }
}

TEST(ScalarMove, InvalidFlagsEvenWhenConditionIsFalse) {
  for (unsigned bit = 0; bit < 64; ++bit) {
    uint32_t d = 17;
    uint64_t d64 = 29;
    uint64_t mode = 1ULL << bit;
    EXPECT_EQ(goc_s_mov_b32(0, mode, &d, 0), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_s_mov_b64(0, mode, &d64, 0), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_s_cmov_b32(0, mode, &d, 0, 0), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_s_cmov_b64(0, mode, &d64, 0, 0), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_s_movk_i32(0, mode, &d, 0), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_s_cmovk_i32(0, mode, &d, 0, 0), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(d, 17U);
    EXPECT_EQ(d64, 29U);
  }
}

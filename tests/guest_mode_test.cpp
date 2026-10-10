// SPDX-License-Identifier: MIT

#include "fp_environment.h"
#include "goc/goc.h"

#include <cfenv>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdint.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace {

const uint64_t exact = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;

} // namespace

TEST(GuestMode, EveryImmediatePreservesUnselectedBits) {
  for (uint64_t semantics : std::initializer_list<uint64_t>{0, exact})
    for (uint32_t immediate = 0; immediate < 65536; ++immediate) {
      uint32_t initial = immediate * 0x1234567U + 0xdeadbeefU;
      uint32_t round = initial, denorm = initial;
      ASSERT_EQ(goc_s_round_mode(semantics, 0, immediate, &round), GOC_SUCCESS);
      ASSERT_EQ(goc_s_denorm_mode(semantics, 0, immediate, &denorm), GOC_SUCCESS);
      EXPECT_EQ(round & 15, immediate & 15);
      EXPECT_EQ(round >> 4, initial >> 4);
      EXPECT_EQ((denorm >> 4) & 15, immediate & 15);
      EXPECT_EQ(denorm & 0xffffff0fU, initial & 0xffffff0fU);
      ASSERT_EQ(goc_s_denorm_mode(semantics, 0, immediate, &round), GOC_SUCCESS);
      EXPECT_EQ(round, (initial & 0xffffff00U) | ((immediate & 15) * 17));
    }
}

TEST(GuestMode, NullOptOutAndErrorPreservation) {
  for (auto fn : {goc_s_round_mode, goc_s_denorm_mode}) {
    for (uint64_t semantics : std::initializer_list<uint64_t>{0, exact})
      EXPECT_EQ(fn(semantics, 0, 65535, nullptr), GOC_SUCCESS);
    uint32_t mode = 0x12345678;
    for (unsigned bit = 0; bit < 64; ++bit) {
      EXPECT_EQ(fn(exact, 1ULL << bit, 15, &mode), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(exact, 1ULL << bit, 15, nullptr), GOC_ERROR_INVALID_FLAGS);
    }
    EXPECT_EQ(fn(1ULL << 63, 0, 15, &mode), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(fn(2 * GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 15, &mode),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    EXPECT_EQ(mode, 0x12345678U);
  }
}

TEST(GuestMode, PreservesHostRoundingExceptionFlagsAndControls) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    ASSERT_EQ(std::fesetround(rounding), 0);
    ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
    ASSERT_EQ(std::feraiseexcept(FE_INVALID | FE_INEXACT), 0);
    int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
#if defined(__x86_64__) || defined(_M_X64)
    unsigned controls = _mm_getcsr();
#endif
    for (uint16_t immediate = 0; immediate < 16; ++immediate) {
      uint32_t mode = 0;
      ASSERT_EQ(goc_s_round_mode(exact, 0, immediate, &mode), GOC_SUCCESS);
      ASSERT_EQ(goc_s_denorm_mode(exact, 0, immediate, &mode), GOC_SUCCESS);
      EXPECT_EQ(std::fegetround(), rounding);
      EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
#if defined(__x86_64__) || defined(_M_X64)
      EXPECT_EQ(_mm_getcsr(), controls);
#endif
    }
  }
}

TEST(GuestMode, GetpcAdvancesWithoutChangingInputAndAllowsAliasing) {
  for (uint64_t pc : std::initializer_list<uint64_t>{0, 4, 0x123456789abcULL, 0x7ffffffffffcULL,
                                                     UINT64_MAX - 3}) {
    uint64_t d = 0;
    ASSERT_EQ(goc_s_getpc_b64(exact, 0, &d, pc), GOC_SUCCESS);
    EXPECT_EQ(d, pc + 4);
    d = pc;
    ASSERT_EQ(goc_s_getpc_b64(exact, 0, &d, d), GOC_SUCCESS);
    EXPECT_EQ(d, pc + 4);
  }
  uint64_t d = 0xdeadbeef;
  EXPECT_EQ(goc_s_getpc_b64(exact, 1, &d, 0), GOC_ERROR_INVALID_FLAGS);
  EXPECT_EQ(d, 0xdeadbeefU);
}

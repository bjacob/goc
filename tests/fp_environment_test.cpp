// SPDX-License-Identifier: MIT

#include "fp_environment.h"

#include <cfenv>
#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace {

void fail_after_changing_environment() {
  goc_test::ScopedFpEnvironment restore;
  ASSERT_TRUE(restore.saved());
  ASSERT_EQ(std::fesetround(FE_UPWARD), 0);
  ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
  ASSERT_EQ(std::feraiseexcept(FE_INVALID), 0);
#if defined(__x86_64__) || defined(_M_X64)
  _mm_setcsr(_mm_getcsr() ^ 0x8040u);
#endif
  FAIL() << "intentional early exit";
}

} // namespace

TEST(FpEnvironment, RestoresAfterFatalAssertion) {
  goc_test::ScopedFpEnvironment original;
  ASSERT_TRUE(original.saved());
  ASSERT_EQ(std::fesetround(FE_DOWNWARD), 0);
  ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
  ASSERT_EQ(std::feraiseexcept(FE_DIVBYZERO), 0);
#if defined(__x86_64__) || defined(_M_X64)
  const unsigned mxcsr = _mm_getcsr();
#endif
  EXPECT_FATAL_FAILURE(fail_after_changing_environment(), "intentional early exit");
  EXPECT_EQ(std::fegetround(), FE_DOWNWARD);
  EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
#if defined(__x86_64__) || defined(_M_X64)
  EXPECT_EQ(_mm_getcsr(), mxcsr);
#endif
}

TEST(FpEnvironment, NestedScopesRestoreTheirOwnState) {
  goc_test::ScopedFpEnvironment original;
  ASSERT_TRUE(original.saved());
  ASSERT_EQ(std::fesetround(FE_TOWARDZERO), 0);
  ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
  {
    goc_test::ScopedFpEnvironment outer;
    ASSERT_TRUE(outer.saved());
    ASSERT_EQ(std::fesetround(FE_DOWNWARD), 0);
    ASSERT_EQ(std::feraiseexcept(FE_DIVBYZERO), 0);
    {
      goc_test::ScopedFpEnvironment inner;
      ASSERT_TRUE(inner.saved());
      ASSERT_EQ(std::fesetround(FE_UPWARD), 0);
      ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
    }
    EXPECT_EQ(std::fegetround(), FE_DOWNWARD);
    EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
  }
  EXPECT_EQ(std::fegetround(), FE_TOWARDZERO);
  EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), 0);
}

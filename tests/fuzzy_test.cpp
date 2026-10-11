// SPDX-License-Identifier: MIT

#include "fuzzy.h"

#include <gtest/gtest.h>
#include <limits>
#include <string>

TEST(Fuzzy, AbsoluteAndRelativeBounds) {
  EXPECT_TRUE(goc_test::fuzzy(0.25, 0, {0.25, 0}));
  EXPECT_FALSE(goc_test::fuzzy(0.250001, 0, {0.25, 0}));
  EXPECT_TRUE(goc_test::fuzzy(1.5, 1, {0.25, 0.25}));
  EXPECT_FALSE(goc_test::fuzzy(1.500001, 1, {0.25, 0.25}));
  EXPECT_TRUE(goc_test::fuzzy(-1.5, -1, {0.25, 0.25}));
  EXPECT_FALSE(goc_test::fuzzy(2, 1, {0, 0.6}));
  EXPECT_TRUE(goc_test::fuzzy(1, 2, {0, 0.6}));
}

TEST(Fuzzy, SpecialValuesAndExactBounds) {
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(goc_test::fuzzy(nan, nan, {0, 0}));
  EXPECT_TRUE(goc_test::fuzzy(inf, inf, {0, 0}));
  EXPECT_TRUE(goc_test::fuzzy(-inf, -inf, {0, 0}));
  EXPECT_TRUE(goc_test::fuzzy(-0., 0., {0, 0}));
  EXPECT_TRUE(goc_test::fuzzy(1, 1, {0, 0}));
  EXPECT_FALSE(goc_test::fuzzy(1, 1.000001, {0, 0}));
  EXPECT_FALSE(goc_test::fuzzy(nan, 0, {1, 1}));
  EXPECT_FALSE(goc_test::fuzzy(0, nan, {1, 1}));
  EXPECT_FALSE(goc_test::fuzzy(inf, -inf, {1, 1}));
  EXPECT_FALSE(goc_test::fuzzy(inf, 1, {1, 1}));
  EXPECT_FALSE(goc_test::fuzzy(1, inf, {1, 1}));
  // An overflowing bound cannot make a finite/infinite pair match.
  EXPECT_FALSE(goc_test::fuzzy(inf, 1e308, {1e308, 2}));
}

TEST(Fuzzy, InvalidBoundsAndFailureDiagnostics) {
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (goc_test::Tolerance tolerance : {goc_test::Tolerance{-1, 0}, {0, -1}, {nan, 0}, {0, inf}})
    EXPECT_FALSE(goc_test::fuzzy(1, 1, tolerance));
  const auto failure = goc_test::fuzzy(3, 1, {0.25, 0.5});
  ASSERT_FALSE(failure);
  const std::string message = failure.message();
  for (const char *part :
       {"actual=3", "expected=1", "abs_error=2", "allowed=0.75", "atol=0.25", "rtol=0.5"})
    EXPECT_NE(message.find(part), std::string::npos);
}

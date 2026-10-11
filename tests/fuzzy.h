// SPDX-License-Identifier: MIT

#pragma once

#include "fuzzy_compare.h"

#include <cmath>
#include <gtest/gtest.h>
#include <iomanip>

namespace goc_test {

// Compare with |actual-expected| <= atol + rtol*|expected|. Both tolerances
// must be finite and nonnegative. NaNs match NaNs; equal infinities and signed
// zeros match. Other nonfinite pairs fail. The relative term is asymmetric.
// This consolidates GoC's WMMA comparator, following the NumPy convention also
// used by hrx-system/iree's iree_math_fuzzy_compare_f64.
inline ::testing::AssertionResult fuzzy(double actual, double expected, Tolerance tolerance) {
  if (!std::isfinite(tolerance.atol) || !std::isfinite(tolerance.rtol) || tolerance.atol < 0 ||
      tolerance.rtol < 0)
    return ::testing::AssertionFailure()
           << "invalid tolerances: atol=" << tolerance.atol << ", rtol=" << tolerance.rtol;
  if (fuzzy_equal(actual, expected, tolerance))
    return ::testing::AssertionSuccess();
  const double error = std::abs(actual - expected);
  const double bound = tolerance.atol + tolerance.rtol * std::abs(expected);
  return ::testing::AssertionFailure()
         << std::setprecision(17) << "actual=" << actual << ", expected=" << expected
         << ", abs_error=" << error << ", allowed=" << bound << " (atol=" << tolerance.atol
         << ", rtol=" << tolerance.rtol << ")";
}

} // namespace goc_test

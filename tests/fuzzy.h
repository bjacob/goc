// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <gtest/gtest.h>
#include <iomanip>

namespace goc_test {

struct Tolerance {
  double atol;
  double rtol;
};

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
  if (actual == expected || (std::isnan(actual) && std::isnan(expected)))
    return ::testing::AssertionSuccess();
  const double error = std::abs(actual - expected);
  const double bound = tolerance.atol + tolerance.rtol * std::abs(expected);
  if (std::isfinite(actual) && std::isfinite(expected) && error <= bound)
    return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure()
         << std::setprecision(17) << "actual=" << actual << ", expected=" << expected
         << ", abs_error=" << error << ", allowed=" << bound << " (atol=" << tolerance.atol
         << ", rtol=" << tolerance.rtol << ")";
}

} // namespace goc_test

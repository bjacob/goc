// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>

namespace goc_test {

struct Tolerance {
  double atol;
  double rtol;
};

// NumPy-style asymmetric comparison. NaNs match NaNs, signed zeros compare
// equal, and infinities must match exactly. Invalid tolerances always fail.
// Independent of GoogleTest so benchmark correctness checks use the same rule.
inline bool fuzzy_equal(double actual, double expected, Tolerance tolerance) {
  if (!std::isfinite(tolerance.atol) || !std::isfinite(tolerance.rtol) || tolerance.atol < 0 ||
      tolerance.rtol < 0)
    return false;
  if (actual == expected || (std::isnan(actual) && std::isnan(expected)))
    return true;
  return std::isfinite(actual) && std::isfinite(expected) &&
         std::abs(actual - expected) <= tolerance.atol + tolerance.rtol * std::abs(expected);
}

} // namespace goc_test

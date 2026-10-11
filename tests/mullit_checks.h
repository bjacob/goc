// SPDX-License-Identifier: MIT

#pragma once

#include "fuzzy.h"
#include "loose_tolerances.h"
#include "mullit_reference.h"

#include <gtest/gtest.h>
#include <stdint.h>

namespace goc_test {

// Compare loose numerical results while preserving the lighting sentinel and
// signed-zero contract.
inline ::testing::AssertionResult mullit_result(uint32_t actual, uint32_t expected) {
  if ((expected & 0x7fffffffU) == 0 || expected == 0xff7fffffU) {
    if (actual == expected)
      return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << "bits=" << actual << ", expected=" << expected;
  }
  return fuzzy(mullit_float(actual), mullit_float(expected), mullit_tolerance);
}

} // namespace goc_test

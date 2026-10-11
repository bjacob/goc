// SPDX-License-Identifier: MIT

#pragma once

#include "fuzzy.h"
#include "goc/goc.h"
#include "half_reference.h"
#include "loose_tolerances.h"

#include <gtest/gtest.h>
#include <stdint.h>

namespace goc_test {

// Compare a written half result, preserving the other half exactly. Arithmetic
// may differ within its loose tolerance; selection and categorical results stay raw.
inline ::testing::AssertionResult half_binary_equal(uint32_t actual, uint32_t expected,
                                                    uint32_t mode, bool arithmetic) {
  unsigned shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  if ((actual & ~(0xffffU << shift)) != (expected & ~(0xffffU << shift)))
    return ::testing::AssertionFailure() << "unselected destination half changed";
  uint16_t got = uint16_t(actual >> shift), want = uint16_t(expected >> shift);
  unsigned magnitude = want & 0x7fff;
  if (magnitude > 0x7c00) {
    if ((got & 0x7fff) > 0x7c00)
      return ::testing::AssertionSuccess();
  } else if (arithmetic && magnitude && magnitude < 0x7c00) {
    return fuzzy(half_value(got), half_value(want), scalar_arithmetic_f16_tolerance);
  } else if (got == want) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << "actual half=" << got << " expected half=" << want;
}

} // namespace goc_test

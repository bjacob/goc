// SPDX-License-Identifier: MIT

#pragma once

#include "fuzzy.h"

namespace goc_test {

// Existing bounds, centralized without widening. Empirical calibration and
// domain coverage are tracked in TESTING_AUDIT.md; these are not universal
// tolerances for every instruction producing the same numeric format.
inline const Tolerance wmma_f32_tolerance = {1e-5, 1e-5};
inline const Tolerance interp_f32_tolerance = {1e-44, 2e-7};
inline const Tolerance trig_f32_tolerance = {3e-7, 0};
inline const Tolerance trig_f16_tolerance = {0.0005, 0};

} // namespace goc_test

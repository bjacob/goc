// SPDX-License-Identifier: MIT

#pragma once

#include "fuzzy_compare.h"

namespace goc_test {

// Family/domain-specific bounds. Measurements and provenance are recorded in
// LOOSE_TOLERANCES.md; these are not universal tolerances for every instruction
// producing the same numeric format.
// Half-output DOT2 with BF16 operands/results; measured bounded domain.
inline const Tolerance half_dot_bf16_tolerance = {0x1p-126, 4e-3};
inline const Tolerance half_sfu_tolerance = {0x1p-24, 1.1e-3};
inline const Tolerance fp8_dot_repeated_tolerance = {0, 2.5e-7};
// FP16/BF16 factors bounded by 4 with a bounded FP32 accumulator.
inline const Tolerance dot2_bounded_tolerance = {2e-6, 2e-7};
// Mixed FP8/BF8 factors bounded by 4, including cancellation against C.
inline const Tolerance fp8_dot_bounded_tolerance = {4e-6, 2e-7};
inline const Tolerance scalar_arithmetic_f16_tolerance = {0x1p-24, 1e-3};
inline const Tolerance scalar_arithmetic_f32_tolerance = {0x1p-149, 1.2e-7};
inline const Tolerance arithmetic_f64_tolerance = {0x1p-1074, 2.5e-16};
inline const Tolerance unary_f64_tolerance = {0x1p-1072, 5e-16};
inline const Tolerance hardware_sfu_f64_tolerance = {0, 0x1p-23};
inline const Tolerance unary_f32_tolerance = {0, 2.5e-7};
inline const Tolerance fma_f16_tolerance = {0x1p-24, 1e-3};
inline const Tolerance fma_f32_tolerance = {0x1p-149, 1.2e-7};
inline const Tolerance mullit_tolerance = {0x1p-149, 1.2e-7};
inline const Tolerance wmma_f32_tolerance = {1e-5, 1e-5};
inline const Tolerance interp_f32_tolerance = {1e-44, 2e-7};
inline const Tolerance trig_f32_tolerance = {3e-7, 0};
// Captured FP32 architecture corpus including output scaling.
inline const Tolerance trig_arch_f32_tolerance = {8e-7, 0};
inline const Tolerance trig_arch_subnormal_f32_tolerance = {3 * 0x1p-149, 0};
inline const Tolerance trig_f16_tolerance = {0.0005, 0};

} // namespace goc_test

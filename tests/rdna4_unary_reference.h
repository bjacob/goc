// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"
#include "internal.h"
#include "rdna4_omod_reference.h"

#include <algorithm>
#include <cmath>
#include <stdint.h>

namespace goc_test {

inline const char *const unary_names[] = {
    "v_trunc_f32", "v_ceil_f32", "v_rndne_f32", "v_floor_f32", "v_sqrt_f32",      "v_rcp_f32",
    "v_rsq_f32",   "v_exp_f32",  "v_log_f32",   "v_fract_f32", "v_frexp_mant_f32"};

using UnaryFn = decltype(&goc_rdna4_v_log_f32);
inline const UnaryFn unary_functions[] = {
    goc_rdna4_v_trunc_f32, goc_rdna4_v_ceil_f32,  goc_rdna4_v_rndne_f32,     goc_rdna4_v_floor_f32,
    goc_rdna4_v_sqrt_f32,  goc_rdna4_v_rcp_f32,   goc_rdna4_v_rsq_f32,       goc_rdna4_v_exp_f32,
    goc_rdna4_v_log_f32,   goc_rdna4_v_fract_f32, goc_rdna4_v_frexp_mant_f32};

// Independent higher-precision reference, with explicit ties-to-even and
// binary32 rounding before output scaling.
inline float unary_reference(int op, float input, uint32_t flags) {
  double x = input;
  if (flags & GOC_ALU_ABS_A)
    x = std::abs(x);
  if (flags & GOC_ALU_NEG_A)
    x = -x;
  bool flush = op >= 4 && op <= 8;
  if (flush && std::abs(x) < std::ldexp(1.0, -126))
    x = std::copysign(0.0, x);
  double y = 0;
  switch (op) {
  case 10: {
    int exponent;
    y = std::isfinite(x) ? std::frexp(x, &exponent) : x;
    break;
  }
  case 0:
    y = std::trunc(x);
    break;
  case 1:
    y = std::ceil(x);
    break;
  case 2:
    y = x;
    if (std::isfinite(x) && x != 0) {
      double lo = std::floor(x), fraction = x - lo;
      y = lo + (fraction > 0.5 || (fraction == 0.5 && std::fmod(lo, 2) != 0));
      y = std::copysign(std::abs(y), x);
    }
    break;
  case 3:
    y = std::floor(x);
    break;
  case 4:
    y = std::sqrt(x);
    break;
  case 5:
    y = 1 / x;
    break;
  case 6:
    y = 1 / std::sqrt(x);
    break;
  case 7:
    y = std::exp2(x);
    break;
  case 9:
    y = x - std::floor(x);
    if (y > double(goc::as_float(0x3f7fffff)))
      y = goc::as_float(0x3f7fffff);
    break;
  case 8:
    y = std::log2(x);
    break;
  }
  float result = float(y);
  if (flush && std::abs(result) < std::ldexp(1.0f, -126))
    result = std::copysign(0.0f, result);
  result = goc_test::omod_f32_reference(result, flags);
  if (flags & GOC_ALU_CLAMP)
    result = std::isnan(result) || result <= 0 ? 0 : std::min(result, 1.0f);
  return result;
}

} // namespace goc_test

// SPDX-License-Identifier: MIT

#pragma once

#include "float_formats.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_fp64.h"

#include <cfenv>
#include <cmath>
#include <limits>
#include <stdint.h>

namespace goc {

// Restore rounding, exception flags and trap enables after exact arithmetic.
class HalfFmaEnvironment {
public:
  explicit HalfFmaEnvironment(bool exact) : active(exact) {
    if (active) {
      std::feholdexcept(&saved);
      std::fesetround(FE_TONEAREST);
    }
  }

  ~HalfFmaEnvironment() {
    if (active)
      std::fesetenv(&saved);
  }

private:
  bool active;
  std::fenv_t saved;
};

// Round finite double values to FP16 without double rounding through FP32.
inline uint16_t half_fma_narrow(double value, bool saturate) {
  float rounded = float(value);
  uint32_t bits = goc::as_bits(rounded);
  if (double(rounded) != value && !(bits & 1)) {
    bool increase = (value > double(rounded)) == !std::signbit(rounded);
    rounded = goc::as_float(bits + (increase ? 1u : UINT32_MAX));
  }
  return goc::float_to_f16(rounded, saturate);
}

inline uint16_t half_fma_clamp(uint16_t value) {
  return (value & 0x8000) || (value & 0x7fff) > 0x7c00 ? 0 : value > 0x3c00 ? 0x3c00 : value;
}

// Adapted from rocjitsu shared/fp_mode.h: fma_f16 and finish_fma_f16.
// RNE and preserved input/output denormals are the currently exposed FP policy.
inline uint16_t half_fma_value(uint16_t a, uint16_t b, uint16_t c, uint32_t mode, bool saturate) {
  uint16_t inputs[] = {a, b, c};
  for (int i = 0; i < 3; ++i) {
    if (mode & (GOC_ALU_ABS_A << i))
      inputs[i] &= 0x7fff;
    if (mode & (GOC_ALU_NEG_A << i))
      inputs[i] ^= 0x8000;
  }
  a = inputs[0];
  b = inputs[1];
  c = inputs[2];
  uint16_t ma = a & 0x7fff, mb = b & 0x7fff, mc = c & 0x7fff;
  uint16_t exceptional = 0;
  if ((ma == 0 && mb == 0x7c00) || (mb == 0 && ma == 0x7c00))
    exceptional = 0xfe00;
  else if (ma > 0x7c00 || mb > 0x7c00 || mc > 0x7c00)
    exceptional = (ma > 0x7c00 ? a : mb > 0x7c00 ? b : c) | 0x0200;
  else if (ma == 0x7c00 || mb == 0x7c00) {
    exceptional = ((a ^ b) & 0x8000) | 0x7c00;
    if (mc == 0x7c00 && ((exceptional ^ c) & 0x8000))
      exceptional = 0xfe00;
  } else if (mc == 0x7c00)
    exceptional = c;
  if (exceptional)
    return mode & GOC_ALU_CLAMP ? half_fma_clamp(exceptional) : exceptional;

  // Half products are exact in double. TwoSum plus round-to-odd retains tiny
  // addends that would otherwise disappear at a later half rounding boundary.
  double product = double(goc::f16_to_float(a)) * double(goc::f16_to_float(b));
  double addend = goc::f16_to_float(c);
  double value = product + addend;
  double virtual_c = value - product;
  double error = (product - (value - virtual_c)) + (addend - virtual_c);
  if (error != 0 && !(goc::double_bits(value) & 1))
    value = std::nextafter(value, std::copysign(std::numeric_limits<double>::infinity(), error));
  // Under nearest-even, only matching negative zero terms produce -0.
  if (value == 0)
    value = ((a ^ b) & c & 0x8000) && mc == 0 ? -0.0 : 0.0;

  uint16_t result = half_fma_narrow(value, saturate);
  uint32_t omod = (mode >> 6) & 3;
  if (omod) {
    uint16_t magnitude = result & 0x7fff;
    if (magnitude < 0x0400 || (magnitude == 0x0400 && std::abs(value) < 0x1p-14 - 0x1p-26))
      result = 0;
    else if (omod == 3 && magnitude < 0x0800)
      result &= 0x8000;
    else {
      const float scales[] = {1, 2, 4, 0.5f};
      result = goc::float_to_f16(goc::f16_to_float(result) * scales[omod], saturate);
    }
  }
  return mode & GOC_ALU_CLAMP ? half_fma_clamp(result) : result;
}

} // namespace goc

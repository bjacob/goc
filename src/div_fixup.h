// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Adapted from rocjitsu shared/division.h; FP16 saturation and OMOD behavior
// follow RX 9070 captures rather than the promoted-FP32 rocjitsu implementation.

#pragma once

#include "division.h"
#include "goc/goc.h"

#include <stdint.h>

namespace goc {

// Repair provisional quotient P using denominator B and numerator C, preserving
// denormals and applying nearest-even overflow rules. Applies all source/output
// modifiers and FP16 saturation without reading or changing host FP state.
template <unsigned Width>
inline typename DivisionFormat<Width>::Bits
fixup_value(typename DivisionFormat<Width>::Bits p, typename DivisionFormat<Width>::Bits b,
            typename DivisionFormat<Width>::Bits c, uint32_t mode, bool saturate) {
  using F = DivisionFormat<Width>;
  using T = typename F::Bits;
  T inputs[] = {p, b, c};
  for (unsigned i = 0; i < 3; ++i) {
    if (mode & (GOC_ALU_ABS_A << i))
      inputs[i] &= F::sign - 1;
    if (mode & (GOC_ALU_NEG_A << i))
      inputs[i] ^= F::sign;
  }
  p = inputs[0];
  b = inputs[1];
  c = inputs[2];
  T bm = b & (F::sign - 1), cm = c & (F::sign - 1), pm = p & (F::sign - 1),
    sign = (b ^ c) & F::sign, result;
  T overflow = F::infinity - (Width == 16 && saturate);
  if (cm > F::infinity)
    result = c | F::quiet;
  else if (bm > F::infinity)
    result = b | F::quiet;
  else if ((bm == 0 && cm == 0) || (bm == F::infinity && cm == F::infinity))
    result = F::sign | F::infinity | F::quiet;
  else if (bm == 0 || cm == F::infinity)
    result = sign | F::infinity;
  else if (cm == 0 || bm == F::infinity)
    result = sign;
  else if (Width != 16 &&
           int(cm >> F::fraction) - int(bm >> F::fraction) < -(F::bias + F::fraction))
    result = sign;
  else
    result = sign | (pm >= F::infinity ? overflow : pm);
  return division_output<Width>(result, mode, overflow);
}

// Classify DIV_FIXUP's wave exception contribution from unmodified source bits.
// Adapted from rocjitsu shared/alu_exceptions.h, with RX 9070 corrections:
// CLAMP suppresses all flags; NaN/invalid cases suppress input-denormal flags;
// FP16 has no exponent-gap underflow recovery. Signs do not affect these flags.
template <unsigned Width>
inline uint32_t fixup_exceptions(typename DivisionFormat<Width>::Bits p,
                                 typename DivisionFormat<Width>::Bits b,
                                 typename DivisionFormat<Width>::Bits c, uint32_t mode) {
  using F = DivisionFormat<Width>;
  p &= F::sign - 1;
  b &= F::sign - 1;
  c &= F::sign - 1;
  if (mode & GOC_ALU_CLAMP)
    return 0;
  if ((b > F::infinity && !(b & F::quiet)) || (c > F::infinity && !(c & F::quiet)))
    return GOC_EXCEPTION_INVALID;
  if (b > F::infinity || c > F::infinity)
    return 0;
  if ((!b && !c) || (b == F::infinity && c == F::infinity))
    return GOC_EXCEPTION_INVALID;
  if (!b && c && c < F::infinity)
    return GOC_EXCEPTION_FLOAT_DIV0;
  uint32_t exceptions = ((b && b < (1ULL << F::fraction)) || (c && c < (1ULL << F::fraction)))
                            ? GOC_EXCEPTION_INPUT_DENORM
                            : 0;
  if (!b || !c || b >= F::infinity || c >= F::infinity)
    return exceptions;
  unsigned omod = (mode >> 6) & 3;
  if (Width != 16 && int(c >> F::fraction) - int(b >> F::fraction) < -(F::bias + F::fraction))
    return exceptions | (omod ? 0 : GOC_EXCEPTION_UNDERFLOW | GOC_EXCEPTION_INEXACT);
  if (p >= F::infinity)
    return exceptions | GOC_EXCEPTION_OVERFLOW | (omod ? 0 : GOC_EXCEPTION_INEXACT);
  if ((omod == 1 || omod == 2) && (p >> F::fraction) + omod >= unsigned(2 * F::bias + 1))
    exceptions |= GOC_EXCEPTION_OVERFLOW;
  return exceptions;
}

template <unsigned Width>
void fixup_x86_64_v3(uint32_t exec_mask, uint32_t mode, bool saturate, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c);

template <unsigned Width>
void fixup_x86_64_v4(uint32_t exec_mask, uint32_t mode, bool saturate, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c);

} // namespace goc

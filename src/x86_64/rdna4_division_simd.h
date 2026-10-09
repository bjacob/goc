// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"
#include "rdna4_division.h"

#include <stdint.h>

namespace goc {

// Apply division OMOD and CLAMP to raw encodings, using the supplied overflow value.
template <unsigned Width, class O>
inline typename O::V division_vector_output(typename O::V result, uint32_t mode,
                                            typename O::V overflow) {
  using F = DivisionFormat<Width>;
  auto zero = O::set(0), inf = O::set(F::infinity), signbit = O::set(F::sign);
  unsigned omod = (mode >> 6) & 3;
  if (omod) {
    auto exponent = O::band(result, inf), output_sign = O::band(result, signbit),
         step = O::set(uint64_t(1) << F::fraction);
    typename O::V scaled;
    if (omod == 3)
      scaled = O::select(O::eq(exponent, step), output_sign, O::sub(result, step));
    else {
      auto increment = O::set(uint64_t(omod) << F::fraction);
      scaled = O::select(O::inverse(O::gt(O::sub(inf, increment), exponent)),
                         O::bor(output_sign, overflow), O::add(result, increment));
    }
    result = O::select(O::eq(exponent, inf), result, scaled);
    result = O::select(O::eq(exponent, zero), zero, result);
  }
  if (mode & GOC_ALU_CLAMP) {
    auto negative = O::inverse(O::eq(O::band(result, signbit), zero));
    result = O::select(O::either(negative, O::gt(result, inf)), zero, result);
    auto one = O::set(uint64_t(F::bias) << F::fraction);
    result = O::select(O::gt(result, one), one, result);
  }
  return result;
}

// Correct the quotient estimate A for exceptional numerator C and denominator B.
template <unsigned Width, class O>
inline typename O::V division_fixup_value(typename O::V a, typename O::V b, typename O::V c,
                                          uint32_t mode, bool saturate) {
  using F = goc::DivisionFormat<Width>;
  auto zero = O::set(0), inf = O::set(F::infinity), signbit = O::set(F::sign),
       magmask = O::set(F::sign - 1);
  auto pm = O::band(a, magmask), bm = O::band(b, magmask), cm = O::band(c, magmask);
  auto sign = O::band(O::bxor(b, c), signbit);
  auto overflow = O::set(F::infinity - (Width == 16 && saturate));
  auto result = O::bor(sign, O::select(O::gt(inf, pm), pm, overflow));
  if constexpr (Width != 16) {
    auto delta = O::sub(O::shr(cm, F::fraction), O::shr(bm, F::fraction));
    result =
        O::select(O::gt(O::set(uint64_t(-int64_t(F::bias + F::fraction))), delta), sign, result);
  }
  auto bz = O::eq(bm, zero), cz = O::eq(cm, zero), bi = O::eq(bm, inf), ci = O::eq(cm, inf);
  result = O::select(O::either(cz, bi), sign, result);
  result = O::select(O::either(bz, ci), O::bor(sign, inf), result);
  result = O::select(O::either(O::both(bz, cz), O::both(bi, ci)),
                     O::set(F::sign | F::infinity | F::quiet), result);
  result = O::select(O::gt(bm, inf), O::bor(b, O::set(F::quiet)), result);
  result = O::select(O::gt(cm, inf), O::bor(c, O::set(F::quiet)), result);
  return division_vector_output<Width, O>(result, mode, overflow);
}

// Scale division inputs and return the per-lane post-scaling condition in post.
template <unsigned Width, class O>
[[gnu::always_inline]] inline typename O::V division_scale_value(typename O::V a, typename O::V b,
                                                                 typename O::V c, uint32_t mode,
                                                                 typename O::Mask &post) {
  using F = goc::DivisionFormat<Width>;
  constexpr int threshold = Width == 32 ? 96 : 768, scale = Width == 32 ? 64 : 128;
  auto zero = O::set(0), one = O::set(1), inf = O::set(F::infinity), signbit = O::set(F::sign),
       magmask = O::set(F::sign - 1);
  auto mag = O::band(a, magmask), bm = O::band(b, magmask), cm = O::band(c, magmask);
  auto de = O::shr(O::band(b, inf), F::fraction), ne = O::shr(O::band(c, inf), F::fraction);
  auto delta = O::sub(ne, de);
  auto denominator = O::eq(a, b);
  auto large_delta = O::gt(delta, O::set(threshold - 1)),
       small_delta = O::gt(O::set(uint64_t(-int64_t(threshold) + 1)), delta);
  post = O::either(large_delta, small_delta);
  auto adjustment = O::select(O::gt(O::set(F::fraction + 2), ne), O::set(scale), zero);
  adjustment = O::select(small_delta, O::select(denominator, zero, O::set(scale)), adjustment);
  auto high_denominator = O::gt(de, O::set(2 * F::bias - 2));
  adjustment = O::select(high_denominator,
                         O::select(O::both(small_delta, O::inverse(denominator)), zero,
                                   O::set(uint64_t(-int64_t(scale)))),
                         adjustment);
  adjustment = O::select(O::eq(de, zero), O::set(scale), adjustment);
  adjustment = O::select(large_delta, O::select(denominator, O::set(scale), zero), adjustment);
  auto fraction_mask = O::set((uint64_t(1) << F::fraction) - 1),
       hidden = O::set(uint64_t(1) << F::fraction);
  auto fraction = O::band(a, fraction_mask);
  // Encoding 2^fraction + significand then subtracting 2^fraction converts
  // the integer significand exactly, without touching subnormal FP operands.
  // The result supplies a normalized mantissa and its leading-bit exponent.
  auto magic = O::set(uint64_t(F::bias + F::fraction) << F::fraction);
  auto encoded = O::bor(magic, fraction);
  auto normalized = O::normalize_fraction(encoded);
  auto original_exponent = O::shr(O::band(a, inf), F::fraction);
  auto subnormal = O::eq(original_exponent, zero);
  auto small_exponent =
      O::sub(O::shr(O::band(normalized, inf), F::fraction), O::set(F::bias + F::fraction - 1));
  auto exponent = O::add(O::select(subnormal, small_exponent, original_exponent), adjustment);
  auto mantissa = O::band(O::select(subnormal, normalized, a), fraction_mask);
  auto sign = O::band(a, signbit);
  auto result = O::bor(sign, O::bor(O::shl(exponent, F::fraction), mantissa));
  auto shift = O::sub(one, exponent);
  shift = O::select(O::gt(one, shift), one, shift);
  shift = O::select(O::gt(shift, O::set(F::fraction + 2)), O::set(F::fraction + 2), shift);
  auto significand = O::bor(mantissa, hidden);
  auto half = O::shlv(one, O::sub(shift, one));
  auto kept = O::shrv(significand, shift);
  auto rounded = O::shrv(O::add(significand, O::add(O::sub(half, one), O::band(kept, one))), shift);
  result = O::select(O::gt(one, exponent), O::bor(sign, rounded), result);
  result = O::select(O::gt(exponent, O::set(2 * F::bias)), O::bor(sign, inf), result);
  auto quiet = O::select(O::gt(mag, inf), O::set(F::quiet), zero);
  result =
      O::select(O::either(O::eq(mag, zero), O::inverse(O::gt(inf, mag))), O::bor(a, quiet), result);
  result = O::select(O::either(O::eq(bm, zero), O::eq(cm, zero)),
                     O::set(F::sign | F::infinity | F::quiet), result);
  return division_vector_output<Width, O>(result, mode, inf);
}

} // namespace goc

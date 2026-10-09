// SPDX-License-Identifier: MIT

#ifndef GOC_RDNA4_INTEGER_TERNARY_H_
#define GOC_RDNA4_INTEGER_TERNARY_H_

#include <stdint.h>

namespace goc {

enum class IntegerTernary { ShiftAdd, AddShift, ShiftOr, AndOr, Or3, Xor3, XorAdd, Lerp };

constexpr bool integer_ternary_v3_supported(IntegerTernary op) {
  return op == IntegerTernary::ShiftAdd || op == IntegerTernary::AddShift ||
         op == IntegerTernary::ShiftOr || op == IntegerTernary::Lerp;
}

template <IntegerTernary Op>
void integer_ternary_x86_64_v3(uint32_t mask, uint32_t *d, const uint32_t *a, const uint32_t *b,
                               const uint32_t *c);

template <IntegerTernary Op>
void integer_ternary_x86_64_v4(uint32_t mask, uint32_t *d, const uint32_t *a, const uint32_t *b,
                               const uint32_t *c);

} // namespace goc

#endif

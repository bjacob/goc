// SPDX-License-Identifier: MIT

#ifndef GOC_TEST_INTEGER_TERNARY_REFERENCE_H_
#define GOC_TEST_INTEGER_TERNARY_REFERENCE_H_

#include "goc/goc.h"

#include <stdint.h>

namespace goc_test {

inline const char *const integer_ternary_names[] = {
    "v_lshl_add_u32", "v_add_lshl_u32", "v_lshl_or_b32", "v_and_or_b32",
    "v_or3_b32",      "v_xor3_b32",     "v_xad_u32",     "v_lerp_u8"};

using IntegerTernaryFn = decltype(&goc_v_lshl_add_u32);
inline const IntegerTernaryFn integer_ternary_functions[] = {
    goc_v_lshl_add_u32, goc_v_add_lshl_u32, goc_v_lshl_or_b32, goc_v_and_or_b32,
    goc_v_or3_b32,      goc_v_xor3_b32,     goc_v_xad_u32,     goc_v_lerp_u8};

// op: shift/add, add/shift, shift/OR, AND/OR, OR3, XOR3, XOR/add, byte LERP.
// Wide arithmetic and per-bit truth tables are independent of the SIMD forms.
inline uint32_t integer_ternary_reference(int op, uint32_t a, uint32_t b, uint32_t c) {
  if (op == 0)
    return uint32_t(uint64_t(a) * (1ULL << (b % 32)) + c);
  if (op == 1)
    return uint32_t((uint64_t(a) + b) * (1ULL << (c % 32)));
  if (op == 7) {
    uint32_t result = 0;
    for (int byte = 0; byte < 4; ++byte) {
      unsigned numerator = (a % 256) + (b % 256) + (c % 2);
      result += (numerator / 2) * (1U << (8 * byte));
      a /= 256;
      b /= 256;
      c /= 256;
    }
    return result;
  }
  uint32_t shifted = op == 2 ? uint32_t(uint64_t(a) * (1ULL << (b % 32))) : a;
  uint32_t result = 0;
  for (int bit = 0; bit < 32; ++bit) {
    unsigned x = (shifted >> bit) & 1, y = (b >> bit) & 1, z = (c >> bit) & 1;
    bool value = op == 2   ? x + z != 0
                 : op == 3 ? x + y == 2 || z != 0
                 : op == 4 ? x + y + z != 0
                 : op == 5 ? (x + y + z) % 2 != 0
                           : x != y;
    if (value)
      result |= 1U << bit;
  }
  return op == 6 ? uint32_t(uint64_t(result) + c) : result;
}

} // namespace goc_test

#endif

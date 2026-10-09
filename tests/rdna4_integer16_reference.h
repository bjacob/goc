// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"
#include "rdna4_packed_integer_reference.h"

#include <stdint.h>

namespace goc_test {

inline uint32_t integer16_reference(int op, uint32_t a, uint32_t b, uint32_t d, uint32_t mode) {
  uint32_t selected_a = mode & GOC_ALU_HIGH_A ? a >> 16 : a & 65535;
  uint32_t selected_b = mode & GOC_ALU_HIGH_B ? b >> 16 : b & 65535;
  uint32_t value = packed_integer_reference(op, selected_a, selected_b,
                                            mode & GOC_ALU_CLAMP ? GOC_PK_CLAMP : 0) &
                   65535;
  return mode & GOC_ALU_HIGH_D ? (d & 65535) | (value << 16) : (d & 0xffff0000u) | value;
}

} // namespace goc_test

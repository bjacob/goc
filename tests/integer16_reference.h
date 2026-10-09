// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"
#include "packed_integer_reference.h"

#include <stdint.h>

namespace goc_test {

inline const char *const integer16_names[] = {"v_add_nc_i16",  "v_sub_nc_i16",  "v_add_nc_u16",
                                              "v_sub_nc_u16",  "v_min_i16",     "v_max_i16",
                                              "v_min_u16",     "v_max_u16",     "v_mul_lo_u16",
                                              "v_lshlrev_b16", "v_lshrrev_b16", "v_ashrrev_i16"};

using Integer16Fn = decltype(&goc_v_add_nc_i16);
inline const Integer16Fn integer16_functions[] = {
    goc_v_add_nc_i16, goc_v_sub_nc_i16,  goc_v_add_nc_u16,  goc_v_sub_nc_u16,
    goc_v_min_i16,    goc_v_max_i16,     goc_v_min_u16,     goc_v_max_u16,
    goc_v_mul_lo_u16, goc_v_lshlrev_b16, goc_v_lshrrev_b16, goc_v_ashrrev_i16};

inline uint32_t integer16_mode_bits(int mode) {
  return (mode & 1 ? GOC_ALU_HIGH_A : 0) | (mode & 2 ? GOC_ALU_HIGH_B : 0) |
         (mode & 4 ? GOC_ALU_HIGH_D : 0) | (mode & 8 ? GOC_ALU_CLAMP : 0);
}

inline uint32_t integer16_reference(int op, uint32_t a, uint32_t b, uint32_t d, uint32_t mode) {
  uint32_t selected_a = mode & GOC_ALU_HIGH_A ? a >> 16 : a & 65535;
  uint32_t selected_b = mode & GOC_ALU_HIGH_B ? b >> 16 : b & 65535;
  uint32_t value = packed_integer_reference(op, selected_a, selected_b,
                                            mode & GOC_ALU_CLAMP ? GOC_PK_CLAMP : 0) &
                   65535;
  return mode & GOC_ALU_HIGH_D ? (d & 65535) | (value << 16) : (d & 0xffff0000u) | value;
}

} // namespace goc_test

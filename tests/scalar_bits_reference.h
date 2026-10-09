// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <stdint.h>

namespace goc_test {

inline const char *const scalar_bits_names[] = {
    "s_and_b32",     "s_and_b64",       "s_or_b32",        "s_or_b64",        "s_xor_b32",
    "s_xor_b64",     "s_nand_b32",      "s_nand_b64",      "s_nor_b32",       "s_nor_b64",
    "s_xnor_b32",    "s_xnor_b64",      "s_and_not1_b32",  "s_and_not1_b64",  "s_or_not1_b32",
    "s_or_not1_b64", "s_not_b32",       "s_not_b64",       "s_brev_b32",      "s_brev_b64",
    "s_lshl_b32",    "s_lshl_b64",      "s_lshr_b32",      "s_lshr_b64",      "s_ashr_i32",
    "s_ashr_i64",    "s_lshl1_add_u32", "s_lshl2_add_u32", "s_lshl3_add_u32", "s_lshl4_add_u32"};

inline int scalar_bits_call(unsigned op, uint64_t flags, uint64_t mode, uint32_t *d, uint64_t *d64,
                            uint64_t a, uint64_t b, uint32_t *scc) {
  switch (op) {
  case 0:
    return goc_s_and_b32(flags, mode, d, uint32_t(a), uint32_t(b), scc);
  case 1:
    return goc_s_and_b64(flags, mode, d64, uint64_t(a), uint64_t(b), scc);
  case 2:
    return goc_s_or_b32(flags, mode, d, uint32_t(a), uint32_t(b), scc);
  case 3:
    return goc_s_or_b64(flags, mode, d64, uint64_t(a), uint64_t(b), scc);
  case 4:
    return goc_s_xor_b32(flags, mode, d, uint32_t(a), uint32_t(b), scc);
  case 5:
    return goc_s_xor_b64(flags, mode, d64, uint64_t(a), uint64_t(b), scc);
  case 6:
    return goc_s_nand_b32(flags, mode, d, uint32_t(a), uint32_t(b), scc);
  case 7:
    return goc_s_nand_b64(flags, mode, d64, uint64_t(a), uint64_t(b), scc);
  case 8:
    return goc_s_nor_b32(flags, mode, d, uint32_t(a), uint32_t(b), scc);
  case 9:
    return goc_s_nor_b64(flags, mode, d64, uint64_t(a), uint64_t(b), scc);
  case 10:
    return goc_s_xnor_b32(flags, mode, d, uint32_t(a), uint32_t(b), scc);
  case 11:
    return goc_s_xnor_b64(flags, mode, d64, uint64_t(a), uint64_t(b), scc);
  case 12:
    return goc_s_and_not1_b32(flags, mode, d, uint32_t(a), uint32_t(b), scc);
  case 13:
    return goc_s_and_not1_b64(flags, mode, d64, uint64_t(a), uint64_t(b), scc);
  case 14:
    return goc_s_or_not1_b32(flags, mode, d, uint32_t(a), uint32_t(b), scc);
  case 15:
    return goc_s_or_not1_b64(flags, mode, d64, uint64_t(a), uint64_t(b), scc);
  case 16:
    return goc_s_not_b32(flags, mode, d, uint32_t(a), scc);
  case 17:
    return goc_s_not_b64(flags, mode, d64, uint64_t(a), scc);
  case 18:
    return goc_s_brev_b32(flags, mode, d, uint32_t(a));
  case 19:
    return goc_s_brev_b64(flags, mode, d64, uint64_t(a));
  case 20:
    return goc_s_lshl_b32(flags, mode, d, uint32_t(a), uint32_t(b), scc);
  case 21:
    return goc_s_lshl_b64(flags, mode, d64, uint64_t(a), uint32_t(b), scc);
  case 22:
    return goc_s_lshr_b32(flags, mode, d, uint32_t(a), uint32_t(b), scc);
  case 23:
    return goc_s_lshr_b64(flags, mode, d64, uint64_t(a), uint32_t(b), scc);
  case 24:
    return goc_s_ashr_i32(flags, mode, d, uint32_t(a), uint32_t(b), scc);
  case 25:
    return goc_s_ashr_i64(flags, mode, d64, uint64_t(a), uint32_t(b), scc);
  case 26:
    return goc_s_lshl1_add_u32(flags, mode, d, uint32_t(a), uint32_t(b), scc);
  case 27:
    return goc_s_lshl2_add_u32(flags, mode, d, uint32_t(a), uint32_t(b), scc);
  case 28:
    return goc_s_lshl3_add_u32(flags, mode, d, uint32_t(a), uint32_t(b), scc);
  case 29:
    return goc_s_lshl4_add_u32(flags, mode, d, uint32_t(a), uint32_t(b), scc);
  }
  return GOC_ERROR_INVALID_FLAGS;
}

} // namespace goc_test

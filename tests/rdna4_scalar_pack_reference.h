// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <stdint.h>

namespace goc_test {

inline const char *const scalar_pack_names[] = {
    "s_pack_ll_b32_b16",      "s_pack_lh_b32_b16", "s_pack_hl_b32_b16", "s_pack_hh_b32_b16",
    "s_bitreplicate_b64_b32", "s_cselect_b32",     "s_cselect_b64",     "s_quadmask_b32",
    "s_quadmask_b64",         "s_wqm_b32",         "s_wqm_b64"};

inline int scalar_pack_call(unsigned op, uint64_t flags, uint64_t mode, uint32_t *d, uint64_t *d64,
                            uint64_t a, uint64_t b, uint32_t *scc, uint32_t input_scc) {
  switch (op) {
  case 0:
    return goc_rdna4_s_pack_ll_b32_b16(flags, mode, d, uint32_t(a), uint32_t(b));
  case 1:
    return goc_rdna4_s_pack_lh_b32_b16(flags, mode, d, uint32_t(a), uint32_t(b));
  case 2:
    return goc_rdna4_s_pack_hl_b32_b16(flags, mode, d, uint32_t(a), uint32_t(b));
  case 3:
    return goc_rdna4_s_pack_hh_b32_b16(flags, mode, d, uint32_t(a), uint32_t(b));
  case 4:
    return goc_rdna4_s_bitreplicate_b64_b32(flags, mode, d64, uint32_t(a));
  case 5:
    return goc_rdna4_s_cselect_b32(flags, mode, d, uint32_t(a), uint32_t(b), input_scc);
  case 6:
    return goc_rdna4_s_cselect_b64(flags, mode, d64, uint64_t(a), uint64_t(b), input_scc);
  case 7:
    return goc_rdna4_s_quadmask_b32(flags, mode, d, uint32_t(a), scc);
  case 8:
    return goc_rdna4_s_quadmask_b64(flags, mode, d64, uint64_t(a), scc);
  case 9:
    return goc_rdna4_s_wqm_b32(flags, mode, d, uint32_t(a), scc);
  case 10:
    return goc_rdna4_s_wqm_b64(flags, mode, d64, uint64_t(a), scc);
  }
  return GOC_ERROR_INVALID_FLAGS;
}

} // namespace goc_test

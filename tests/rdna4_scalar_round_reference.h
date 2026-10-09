// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <stdint.h>

namespace goc_test {

using ScalarRoundFn = decltype(&goc_rdna4_s_ceil_f32);
inline const ScalarRoundFn scalar_round_functions[] = {
    goc_rdna4_s_ceil_f32,  goc_rdna4_s_ceil_f16,  goc_rdna4_s_floor_f32, goc_rdna4_s_floor_f16,
    goc_rdna4_s_trunc_f32, goc_rdna4_s_trunc_f16, goc_rdna4_s_rndne_f32, goc_rdna4_s_rndne_f16};

inline const char *const scalar_round_names[] = {"s_ceil_f32",  "s_ceil_f16",  "s_floor_f32",
                                                 "s_floor_f16", "s_trunc_f32", "s_trunc_f16",
                                                 "s_rndne_f32", "s_rndne_f16"};

inline void scalar_round_inputs(unsigned i, bool half, uint32_t *w) {
  const uint32_t edges[] = {0,          0x80000000, 1,          0x80000001, 0x007fffff, 0x807fffff,
                            0x00800000, 0x80800000, 0x3f000000, 0xbf000000, 0x3fc00000, 0xbfc00000,
                            0x40200000, 0xc0200000, 0x3effffff, 0xbeffffff, 0x3f000001, 0xbf000001,
                            0x3f7fffff, 0xbf7fffff, 0x4affffff, 0xcaffffff, 0x4b000000, 0xcb000000,
                            0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc00001, 0xffc00123,
                            0x7f800001, 0xff800123};
  w[0] = half ? (0xabcd0000u | i) : (i < 32 ? edges[i] : ((i * 0x7395a831u) ^ 0xa7925163u));
  w[1] = 0;
}

} // namespace goc_test

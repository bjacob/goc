// SPDX-License-Identifier: MIT

#pragma once

#include "goc/goc.h"

#include <stdint.h>

namespace goc_test {

inline const char *const scalar_convert_names[] = {
    "s_cvt_f32_i32", "s_cvt_f32_u32", "s_cvt_i32_f32",    "s_cvt_u32_f32",
    "s_cvt_f16_f32", "s_cvt_f32_f16", "s_cvt_hi_f32_f16", "s_cvt_pk_rtz_f16_f32"};

inline void scalar_convert_inputs(unsigned i, unsigned op, uint32_t *w) {
  const uint32_t f[] = {0,          0x80000000, 1,          0x80000001, 0x007fffff, 0x807fffff,
                        0x00800000, 0x80800000, 0x3f000000, 0xbf000000, 0x3fc00000, 0xbfc00000,
                        0x4effffff, 0xceffffff, 0x4f000000, 0xcf000000, 0x4f000001, 0xcf000001,
                        0x4f7fffff, 0xcf7fffff, 0x4f800000, 0xcf800000, 0x477fe000, 0x477ff000,
                        0x387fdfff, 0x387fe000, 0x387fefff, 0x387ff000, 0x7f800000, 0xff800000,
                        0x7f800001, 0xffc00123};
  const uint32_t integer[] = {0,          1,          2,          3,
                              0x7fffffff, 0x80000000, 0xffffffff, 0xfffffffe,
                              0x00ffffff, 0x01000000, 0x01000001, 0x01000002,
                              0x01000003, 0x01ffffff, 0x02000001, 0x02000002};
  if (op == 5) {
    w[0] = ((~i & 65535) << 16) | i;
    w[1] = 0;
    return;
  }
  if (op == 6) {
    w[0] = (i << 16) | (~i & 65535);
    w[1] = 0;
    return;
  }
  w[0] = op < 2 && i < 16    ? integer[i]
         : op >= 2 && i < 32 ? f[i]
                             : ((i * 0x7395a831u) ^ 0xa7925163u);
  w[1] = i < 32 ? f[31 - i] : ((i * 0x9e3779b9u) ^ 0xa5a59669u);
}

inline int scalar_convert_call(unsigned op, uint64_t flags, uint64_t mask, uint64_t mode,
                               uint32_t *d, uint32_t a, uint32_t b) {
  switch (op) {
  case 0:
    return goc_rdna4_s_cvt_f32_i32(flags, mask, mode, d, a);
  case 1:
    return goc_rdna4_s_cvt_f32_u32(flags, mask, mode, d, a);
  case 2:
    return goc_rdna4_s_cvt_i32_f32(flags, mask, mode, d, a);
  case 3:
    return goc_rdna4_s_cvt_u32_f32(flags, mask, mode, d, a);
  case 4:
    return goc_rdna4_s_cvt_f16_f32(flags, mask, mode, d, a);
  case 5:
    return goc_rdna4_s_cvt_f32_f16(flags, mask, mode, d, a);
  case 6:
    return goc_rdna4_s_cvt_hi_f32_f16(flags, mask, mode, d, a);
  case 7:
    return goc_rdna4_s_cvt_pk_rtz_f16_f32(flags, mask, mode, d, a, b);
  }
  return GOC_ERROR_INVALID_FLAGS;
}

} // namespace goc_test

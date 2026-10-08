// SPDX-License-Identifier: MIT
#include "goc.h"
int goc_test_c_api(void) {
  uint32_t a[32] = {0x40000000}, b[32] = {0x40400000}, c[32] = {0x40800000}, d[32] = {0};
  uint32_t *pa = a, *pb = b, *pc = c, *pd = d;
  int status = goc_rdna4_v_fma_f32(0, 1, 0, &pd, &pa, &pb, &pc);
  return status == GOC_SUCCESS && d[0] == 0x41200000;
}

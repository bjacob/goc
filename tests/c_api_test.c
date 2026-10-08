// SPDX-License-Identifier: MIT

#include "goc_common.h"
#include "goc_rdna4.h"

#include <stdint.h>

int goc_test_c_api(void) {
  uint32_t a[32] = {0x40000000}, b[32] = {0x40400000}, c[32] = {0x40800000}, d[32] = {0};
  uint32_t *pa = a, *pb = b, *pc = c, *pd = d;
  int status = goc_rdna4_v_fma_f32(0, 1, 0, &pd, &pa, &pb, &pc);
  if (status != GOC_SUCCESS || d[0] != 0x41200000)
    return 0;

  // Public constants are typed, addressable objects in C99 too. Exercise flag
  // composition and the error contract from C, without relying on C++ rules.
  const uint64_t *exact = &GOC_SEMANTICS_EXACT;
  const uint64_t *strict = &GOC_SEMANTICS_STRICT;
  const int *unsupported = &GOC_ERROR_UNSUPPORTED_SEMANTICS;
  const uint32_t *modifier = &GOC_WMMA_NEG_C;
  status = goc_rdna4_v_fma_f32(*exact | *strict, 1, 0, &pd, &pa, &pb, &pc);
  if (status != *unsupported || d[0] != 0x41200000)
    return 0;
  status = goc_rdna4_v_fma_f32(GOC_CPU_BASELINE, 1, *modifier, &pd, &pa, &pb, &pc);
  return status == GOC_ERROR_INVALID_FLAGS && d[0] == 0x41200000;
}

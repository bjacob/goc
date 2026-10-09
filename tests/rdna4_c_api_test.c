// SPDX-License-Identifier: MIT

#include "goc/goc.h"

#include <stddef.h>
#include <stdint.h>

// Returns a read-only view of VGPR pointers for the C++ implementation.
static inline const uint32_t *const *input_vgprs(uint32_t *const *v) {
  return (const uint32_t *const *)v;
}

int goc_test_c_api(void) {
  int (*fma)(uint64_t, uint32_t, uint64_t, uint32_t *const *, const uint32_t *const *,
             const uint32_t *const *, const uint32_t *const *, uint32_t *) = goc_rdna4_v_fma_f32;
  uint32_t a[32] = {0x40000000}, b[32] = {0x40400000}, c[32] = {0x40800000}, d[32] = {0};
  uint32_t *pa = a, *pb = b, *pc = c, *pd = d;
  int status = fma(0, 1, 0, &pd, input_vgprs(&pa), input_vgprs(&pb), input_vgprs(&pc), NULL);
  if (status != GOC_SUCCESS || d[0] != 0x41200000)
    return 0;

  status =
      goc_rdna4_v_fma_f32(0, 1, 0, &pa, input_vgprs(&pa), input_vgprs(&pb), input_vgprs(&pc), NULL);
  if (status != GOC_SUCCESS || a[0] != 0x41200000)
    return 0;

  uint32_t excp_flag_user = 0x80000021U;
  uint32_t before = d[0];
  if (fma(0, 1, 0, &pd, input_vgprs(&pa), input_vgprs(&pb), input_vgprs(&pc), &excp_flag_user) !=
          GOC_ERROR_UNSUPPORTED_EXCEPTIONS ||
      excp_flag_user != 0x80000021U || d[0] != before)
    return 0;

  // Public constants are typed, addressable objects in C99 too. Exercise flag
  // composition and the error contract from C, without relying on C++ rules.
  const uint64_t *exact = &GOC_SEMANTICS_EXACT_EMPIRICAL;
  const uint64_t *strict = &GOC_SEMANTICS_STRICT;
  const int *unsupported = &GOC_ERROR_UNSUPPORTED_SEMANTICS;
  const uint32_t *modifier = &GOC_ALU_NEG_C;
  status = goc_rdna4_v_fma_f32(*exact | *strict, 1, 0, &pd, input_vgprs(&pa), input_vgprs(&pb),
                               input_vgprs(&pc), NULL);
  if (status != *unsupported || d[0] != 0x41200000)
    return 0;
  status = goc_rdna4_v_fma_f32(GOC_CPU_BASELINE, 1, *modifier, &pd, input_vgprs(&pa),
                               input_vgprs(&pb), input_vgprs(&pc), NULL);
  if (status != GOC_SUCCESS || d[0] != 0x41d00000)
    return 0;
  status = goc_rdna4_v_fma_f32(GOC_CPU_BASELINE, 1, 1U << 31, &pd, input_vgprs(&pa),
                               input_vgprs(&pb), input_vgprs(&pc), NULL);
  if (status != GOC_ERROR_INVALID_FLAGS || d[0] != 0x41d00000)
    return 0;

  status = goc_rdna4_v_fma_f32(0, 1, 1ULL << 63, &pd, input_vgprs(&pa), input_vgprs(&pb),
                               input_vgprs(&pc), NULL);
  if (status != GOC_ERROR_INVALID_FLAGS || d[0] != 0x41d00000)
    return 0;

  // Literal arguments follow assembly order and carry raw floating-point bits.
  const uint32_t *ra = a, *rb = b;
  status = goc_rdna4_v_fmamk_f32(0, 1, 0, &pd, &ra, 0x40000000, &rb, NULL);
  if (status != GOC_SUCCESS || d[0] != 0x41b80000) // 10 * 2 + 3 = 23
    return 0;
  status = goc_rdna4_v_fmaak_f32(0, 1, 0, &pd, &ra, &rb, 0x40000000, NULL);
  if (status != GOC_SUCCESS || d[0] != 0x42000000) // 10 * 3 + 2 = 32
    return 0;
  a[0] = 0x4000;
  b[0] = 0x4200;
  d[0] = 0xfacecafe;
  status = goc_rdna4_v_fmamk_f16(0, 1, 0, &pd, &ra, 0x4400, &rb, NULL);
  if (status != GOC_SUCCESS || d[0] != 0xface4980) // 2 * 4 + 3 = 11
    return 0;
  status = goc_rdna4_v_fmaak_f16(0, 1, 0, &pd, &ra, &rb, 0x4400, NULL);
  return status == GOC_SUCCESS && d[0] == 0xface4900; // 2 * 3 + 4 = 10
}

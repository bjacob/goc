// SPDX-License-Identifier: MIT

// Reciprocal stages follow rocjitsu's transcendental and fp_mode helpers.
// GFX1201 corrections to its exception classifier: flushed subnormals also
// raise INT_DIV0, while CLAMP suppresses a new cause (not an existing flag).

#include "rdna4_rcp_iflag.h"
#include "goc/goc.h"
#include "internal.h"

#include <stdint.h>

int goc_rdna4_v_rcp_iflag_f32(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                              const uint32_t *const *a, uint32_t *exception_flags,
                              uint32_t input_exception_flags) {
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  const uint32_t known = GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP;
  if (int error = goc::validate(flags, mode & ~known, false,
                                GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS))
    return error;
  uint32_t mask = uint32_t(exec_mask);
  if (!mask) {
    *exception_flags = input_exception_flags;
    return GOC_SUCCESS;
  }
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    *exception_flags = input_exception_flags | goc::rcp_iflag_x86_64_v4(mask, mode, d[0], a[0]);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    *exception_flags = input_exception_flags | goc::rcp_iflag_x86_64_v3(mask, mode, d[0], a[0]);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[32], zeros = 0;
  unsigned omod = (mode >> 6) & 3;
  for (unsigned lane = 0; lane < 32; ++lane) {
    uint32_t raw = a[0][lane];
    bool tiny = !(raw & 0x7f800000);
    zeros |= uint32_t(tiny) << lane;
    if (mode & GOC_ALU_ABS_A)
      raw &= 0x7fffffff;
    if (mode & GOC_ALU_NEG_A)
      raw ^= 0x80000000;
    if (tiny)
      raw &= 0x80000000;
    float value = 1.f / goc::as_float(raw);
    uint32_t bits = goc::as_bits(value);
    if (!(bits & 0x7f800000))
      bits &= 0x80000000;
    value = goc::as_float(bits);
    if (omod) {
      if (value == 0)
        value = 0;
      value *= omod == 1 ? 2 : omod == 2 ? 4 : 0.5f;
    }
    if (mode & GOC_ALU_CLAMP)
      value = !(value > 0) ? 0 : value > 1 ? 1 : value;
    bits = goc::as_bits(value);
    if (!(bits & 0x7f800000))
      bits &= 0x80000000;
    result[lane] = bits;
  }
  for (unsigned lane = 0; lane < 32; ++lane)
    if ((mask >> lane) & 1)
      d[0][lane] = result[lane];
  *exception_flags = input_exception_flags |
                     ((zeros & mask) && !(mode & GOC_ALU_CLAMP) ? GOC_RDNA4_EXCEPTION_INT_DIV0 : 0);
  return GOC_SUCCESS;
}

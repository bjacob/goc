// SPDX-License-Identifier: MIT

#include "rdna4_mullit.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_alu.h"
#include "rdna4_dpp.h"

#include <cmath>
#include <stdint.h>

int goc_rdna4_v_mullit_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  if (instruction_flags >> 32)
    return goc::execute_dpp(
        flags, exec_mask, instruction_flags, a, [&](uint64_t mask, const uint32_t *const *source) {
          return goc_rdna4_v_mullit_f32(flags, mask, uint32_t(instruction_flags), d, source, b, c);
        });
  if (int error = goc::validate(flags, instruction_flags & ~UINT32_C(0x1ff)))
    return error;
  if (!uint32_t(exec_mask))
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::mullit_x86_64_v4(uint32_t(exec_mask), instruction_flags, d[0], a[0], b[0], c[0]);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::mullit_x86_64_v3(uint32_t(exec_mask), instruction_flags, d[0], a[0], b[0], c[0]);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (unsigned lane = 0; lane < 32; ++lane) {
    float x = goc::alu_input(a[0][lane], instruction_flags);
    float y = goc::alu_input(b[0][lane], instruction_flags >> 1);
    float z = goc::alu_input(c[0][lane], instruction_flags >> 2);
    uint32_t ybits = goc::as_bits(y);
    float value;
    if (ybits == 0xff7fffff || ybits == 0xff800000 || std::isnan(y) || !(z > 0))
      value = goc::as_float(0xff7fffff);
    else
      value = x == 0 || y == 0 ? 0.0f : x * y;
    unsigned scale = (instruction_flags >> 6) & 3;
    if (scale) {
      // OMOD flushes a tiny unscaled result to +0, then preserves the sign
      // when scaling a normal result produces a tiny result.
      if (!(goc::as_bits(value) & 0x7f800000))
        value = 0;
      value *= scale == 1 ? 2.0f : scale == 2 ? 4.0f : 0.5f;
      uint32_t bits = goc::as_bits(value);
      if (!(bits & 0x7f800000))
        value = goc::as_float(bits & 0x80000000);
    }
    if (instruction_flags & GOC_ALU_CLAMP)
      value = !(value > 0) ? 0 : value > 1 ? 1 : value;
    result[lane] = goc::as_bits(value);
  }
  for (unsigned lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

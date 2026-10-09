// SPDX-License-Identifier: MIT

#include "rdna4_byte_conversion.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_alu.h"

#include <stdint.h>

namespace {

template <unsigned Byte>
int convert(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
            const uint32_t *const *a) {
  if (int error = goc::validate(flags, mode & ~(GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP)))
    return error;
  if (!uint32_t(mask))
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::byte_conversion_x86_64_v4<Byte>(uint32_t(mask), mode, d[0], a[0]);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::byte_conversion_x86_64_v3<Byte>(uint32_t(mask), mode, d[0], a[0]);
    return GOC_SUCCESS;
  }
#endif
  // rocjitsu's execute_v_cvt_f32_ubyte*_vop3 extracts an unsigned byte,
  // converts it exactly, then applies OMOD and CLAMP.
  uint32_t result[32];
  for (unsigned lane = 0; lane < 32; ++lane) {
    float value = float((a[0][lane] >> (8 * Byte)) & 255);
    result[lane] = goc::as_bits(goc::alu_output(value, mode));
  }
  for (unsigned lane = 0; lane < 32; ++lane)
    if ((mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_cvt_f32_ubyte0(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a) {
  return convert<0>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_cvt_f32_ubyte1(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a) {
  return convert<1>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_cvt_f32_ubyte2(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a) {
  return convert<2>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_cvt_f32_ubyte3(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a) {
  return convert<3>(flags, exec_mask, instruction_flags, d, a);
}

// SPDX-License-Identifier: MIT

#include "rdna4_byte_conversion.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_alu.h"
#include "rdna4_dpp.h"

#include <cmath>
#include <stdint.h>

namespace {

template <unsigned Byte>
int convert(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a) {
  if (mode >> 32)
    return goc::execute_dpp(flags, exec_mask, mode, a,
                            [&](uint32_t exec_mask, const uint32_t *const *source) {
                              return convert<Byte>(flags, exec_mask, uint32_t(mode), d, source);
                            });
  if (int error = goc::validate(flags, mode & ~(GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP)))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::byte_conversion_x86_64_v4<Byte>(exec_mask, mode, d[0], a[0]);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::byte_conversion_x86_64_v3<Byte>(exec_mask, mode, d[0], a[0]);
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
    if ((exec_mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_cvt_f32_ubyte0(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a) {
  return convert<0>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_cvt_f32_ubyte1(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a) {
  return convert<1>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_cvt_f32_ubyte2(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a) {
  return convert<2>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_cvt_f32_ubyte3(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a) {
  return convert<3>(flags, exec_mask, instruction_flags, d, a);
}

int goc_rdna4_v_cvt_off_f32_i4(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a) {
  if (mode >> 32)
    return goc::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return goc_rdna4_v_cvt_off_f32_i4(flags, exec_mask, uint32_t(mode), d, source);
        });
  if (int error = goc::validate(flags, mode & ~(GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP)))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::nibble_offset_x86_64_v4(exec_mask, mode, d[0], a[0]);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::nibble_offset_x86_64_v3(exec_mask, mode, d[0], a[0]);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (unsigned lane = 0; lane < 32; ++lane) {
    int nibble = int(a[0][lane] & 15);
    if (nibble & 8)
      nibble -= 16;
    result[lane] = goc::as_bits(goc::alu_output(float(nibble) * 0.0625f, mode));
  }
  for (unsigned lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

int goc_rdna4_v_cvt_pk_u8_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c, uint32_t *excp_flag_user) {
  if (excp_flag_user)
    return GOC_ERROR_UNSUPPORTED_EXCEPTIONS;
  if (mode >> 32)
    return goc::execute_dpp(flags, exec_mask, mode, a,
                            [&](uint32_t exec_mask, const uint32_t *const *source) {
                              return goc_rdna4_v_cvt_pk_u8_f32(flags, exec_mask, uint32_t(mode), d,
                                                               source, b, c, nullptr);
                            });
  if (int error = goc::validate(flags, mode & ~(GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_CLAMP)))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::byte_pack_x86_64_v4(exec_mask, mode, d[0], a[0], b[0], c[0]);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::byte_pack_x86_64_v3(exec_mask, mode, d[0], a[0], b[0], c[0]);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (unsigned lane = 0; lane < 32; ++lane) {
    float value = goc::alu_input(a[0][lane], mode);
    value = !(value > 0) ? 0 : value > 255 ? 255 : value;
    // RX 9070 captures establish nearest-even, not rocjitsu's truncation.
    float lower = std::floor(value), fraction = value - lower;
    uint32_t byte = uint32_t(lower);
    byte += fraction > 0.5f || (fraction == 0.5f && (byte & 1));
    unsigned shift = (b[0][lane] & 3) * 8;
    result[lane] = (c[0][lane] & ~(255u << shift)) | (byte << shift);
  }
  for (unsigned lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

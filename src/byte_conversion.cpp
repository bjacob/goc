// SPDX-License-Identifier: MIT

#include "byte_conversion.h"
#include "alu.h"
#include "dpp.h"
#include "goc/goc.h"
#include "internal.h"

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
  if (int error = goc::validate(flags, mode & ~(GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP), true,
                                GOC_FP_IEEE_MODE | GOC_FP_FLUSH_INPUT_DENORMALS |
                                    GOC_FP_FLUSH_OUTPUT_DENORMALS))
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

int goc_v_cvt_f32_ubyte0_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a) {
  return convert<0>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_cvt_f32_ubyte1_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a) {
  return convert<1>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_cvt_f32_ubyte2_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a) {
  return convert<2>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_cvt_f32_ubyte3_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a) {
  return convert<3>(flags, exec_mask, instruction_flags, d, a);
}

int goc_v_cvt_off_f32_i4_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a) {
  if (mode >> 32)
    return goc::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return goc_v_cvt_off_f32_i4_rdna4(flags, exec_mask, uint32_t(mode), d, source);
        });
  if (int error = goc::validate(flags, mode & ~(GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP), true,
                                GOC_FP_IEEE_MODE | GOC_FP_FLUSH_INPUT_DENORMALS |
                                    GOC_FP_FLUSH_OUTPUT_DENORMALS))
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

int goc_v_cvt_pk_u8_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return goc::execute_dpp(flags, exec_mask, mode, a,
                            [&](uint32_t exec_mask, const uint32_t *const *source) {
                              return goc_v_cvt_pk_u8_f32_rdna4(flags, exec_mask, uint32_t(mode), d,
                                                               source, b, c, nullptr);
                            });
  if (int error = goc::validate(
          flags, mode & ~(GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_CLAMP), false,
          GOC_FP_IEEE_MODE | GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS))
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

namespace {

// Byte 4 denotes the signed-nibble offset conversion.
template <unsigned Byte>
int convert_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a) {
  const auto instruction = Byte == 4 ? goc_v_cvt_off_f32_i4_rdna4 : convert<Byte % 4>;
  if (int error = instruction(flags, 0, mode, nullptr, nullptr))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
  for (unsigned half = 0; half < 2; ++half) {
    uint64_t half_mode = mode;
    if (mode & GOC_DPP16) {
      uint64_t rows = (mode & GOC_DPP_ROW_MASK) >> GOC_DPP_ROW_SHIFT;
      half_mode = (mode & ~GOC_DPP_ROW_MASK) | (((rows >> (half * 2)) & 3) << GOC_DPP_ROW_SHIFT);
    }
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *source = a[0] + half * 32;
    if (int error =
            instruction(flags, uint32_t(exec_mask >> (half * 32)), half_mode, &dest, &source))
      return error;
  }
  return GOC_SUCCESS;
}

} // namespace

int goc_v_cvt_f32_ubyte0_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                      uint32_t *const *d, const uint32_t *const *a) {
  return convert_wave64<0>(flags, exec_mask, mode, d, a);
}

int goc_v_cvt_f32_ubyte1_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                      uint32_t *const *d, const uint32_t *const *a) {
  return convert_wave64<1>(flags, exec_mask, mode, d, a);
}

int goc_v_cvt_f32_ubyte2_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                      uint32_t *const *d, const uint32_t *const *a) {
  return convert_wave64<2>(flags, exec_mask, mode, d, a);
}

int goc_v_cvt_f32_ubyte3_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                      uint32_t *const *d, const uint32_t *const *a) {
  return convert_wave64<3>(flags, exec_mask, mode, d, a);
}

int goc_v_cvt_off_f32_i4_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                      uint32_t *const *d, const uint32_t *const *a) {
  return convert_wave64<4>(flags, exec_mask, mode, d, a);
}

int goc_v_cvt_pk_u8_f32_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, const uint32_t *const *c,
                                     uint32_t *excp_flag_user) {
  if (int error = goc_v_cvt_pk_u8_f32_rdna4(flags, 0, mode, nullptr, nullptr, nullptr, nullptr,
                                            excp_flag_user))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
  for (unsigned half = 0; half < 2; ++half) {
    uint64_t half_mode = mode;
    if (mode & GOC_DPP16) {
      uint64_t rows = (mode & GOC_DPP_ROW_MASK) >> GOC_DPP_ROW_SHIFT;
      half_mode = (mode & ~GOC_DPP_ROW_MASK) | (((rows >> (half * 2)) & 3) << GOC_DPP_ROW_SHIFT);
    }
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *source_a = a[0] + half * 32, *source_b = b[0] + half * 32,
                   *source_c = c[0] + half * 32;
    if (int error =
            goc_v_cvt_pk_u8_f32_rdna4(flags, uint32_t(exec_mask >> (half * 32)), half_mode, &dest,
                                      &source_a, &source_b, &source_c, excp_flag_user))
      return error;
  }
  return GOC_SUCCESS;
}

int goc_v_cvt_f32_ubyte0(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a) {
  // rocjitsu fp_mode::effective_omod: RDNA3 scaling requires output flush and IEEE=0.
  if ((flags & GOC_FP_IEEE_MODE) || !(flags & GOC_FP_FLUSH_OUTPUT_DENORMALS))
    mode &= ~uint64_t(GOC_ALU_OMOD_HALF);
  return goc_v_cvt_f32_ubyte0_rdna4(flags, exec_mask, mode, d, a);
}

int goc_v_cvt_f32_ubyte0_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a) {
  // rocjitsu fp_mode::effective_omod: RDNA3 scaling requires output flush and IEEE=0.
  if ((flags & GOC_FP_IEEE_MODE) || !(flags & GOC_FP_FLUSH_OUTPUT_DENORMALS))
    mode &= ~uint64_t(GOC_ALU_OMOD_HALF);
  return goc_v_cvt_f32_ubyte0_rdna4_wave64(flags, exec_mask, mode, d, a);
}

int goc_v_cvt_f32_ubyte1(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a) {
  // rocjitsu fp_mode::effective_omod: RDNA3 scaling requires output flush and IEEE=0.
  if ((flags & GOC_FP_IEEE_MODE) || !(flags & GOC_FP_FLUSH_OUTPUT_DENORMALS))
    mode &= ~uint64_t(GOC_ALU_OMOD_HALF);
  return goc_v_cvt_f32_ubyte1_rdna4(flags, exec_mask, mode, d, a);
}

int goc_v_cvt_f32_ubyte1_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a) {
  // rocjitsu fp_mode::effective_omod: RDNA3 scaling requires output flush and IEEE=0.
  if ((flags & GOC_FP_IEEE_MODE) || !(flags & GOC_FP_FLUSH_OUTPUT_DENORMALS))
    mode &= ~uint64_t(GOC_ALU_OMOD_HALF);
  return goc_v_cvt_f32_ubyte1_rdna4_wave64(flags, exec_mask, mode, d, a);
}

int goc_v_cvt_f32_ubyte2(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a) {
  // rocjitsu fp_mode::effective_omod: RDNA3 scaling requires output flush and IEEE=0.
  if ((flags & GOC_FP_IEEE_MODE) || !(flags & GOC_FP_FLUSH_OUTPUT_DENORMALS))
    mode &= ~uint64_t(GOC_ALU_OMOD_HALF);
  return goc_v_cvt_f32_ubyte2_rdna4(flags, exec_mask, mode, d, a);
}

int goc_v_cvt_f32_ubyte2_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a) {
  // rocjitsu fp_mode::effective_omod: RDNA3 scaling requires output flush and IEEE=0.
  if ((flags & GOC_FP_IEEE_MODE) || !(flags & GOC_FP_FLUSH_OUTPUT_DENORMALS))
    mode &= ~uint64_t(GOC_ALU_OMOD_HALF);
  return goc_v_cvt_f32_ubyte2_rdna4_wave64(flags, exec_mask, mode, d, a);
}

int goc_v_cvt_f32_ubyte3(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a) {
  // rocjitsu fp_mode::effective_omod: RDNA3 scaling requires output flush and IEEE=0.
  if ((flags & GOC_FP_IEEE_MODE) || !(flags & GOC_FP_FLUSH_OUTPUT_DENORMALS))
    mode &= ~uint64_t(GOC_ALU_OMOD_HALF);
  return goc_v_cvt_f32_ubyte3_rdna4(flags, exec_mask, mode, d, a);
}

int goc_v_cvt_f32_ubyte3_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a) {
  // rocjitsu fp_mode::effective_omod: RDNA3 scaling requires output flush and IEEE=0.
  if ((flags & GOC_FP_IEEE_MODE) || !(flags & GOC_FP_FLUSH_OUTPUT_DENORMALS))
    mode &= ~uint64_t(GOC_ALU_OMOD_HALF);
  return goc_v_cvt_f32_ubyte3_rdna4_wave64(flags, exec_mask, mode, d, a);
}

int goc_v_cvt_off_f32_i4(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a) {
  // rocjitsu fp_mode::effective_omod: RDNA3 scaling requires output flush and IEEE=0.
  if ((flags & GOC_FP_IEEE_MODE) || !(flags & GOC_FP_FLUSH_OUTPUT_DENORMALS))
    mode &= ~uint64_t(GOC_ALU_OMOD_HALF);
  return goc_v_cvt_off_f32_i4_rdna4(flags, exec_mask, mode, d, a);
}

int goc_v_cvt_off_f32_i4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a) {
  // rocjitsu fp_mode::effective_omod: RDNA3 scaling requires output flush and IEEE=0.
  if ((flags & GOC_FP_IEEE_MODE) || !(flags & GOC_FP_FLUSH_OUTPUT_DENORMALS))
    mode &= ~uint64_t(GOC_ALU_OMOD_HALF);
  return goc_v_cvt_off_f32_i4_rdna4_wave64(flags, exec_mask, mode, d, a);
}

int goc_v_cvt_pk_u8_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return goc_v_cvt_pk_u8_f32_rdna4(flags, exec_mask, instruction_flags, d, a, b, c, nullptr);
}

int goc_v_cvt_pk_u8_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c,
                               uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return goc_v_cvt_pk_u8_f32_rdna4_wave64(flags, exec_mask, instruction_flags, d, a, b, c, nullptr);
}

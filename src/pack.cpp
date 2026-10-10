// SPDX-License-Identifier: MIT

#include "pack.h"
#include "dpp.h"
#include "goc/goc.h"
#include "internal.h"

#include <algorithm>
#include <stdint.h>

namespace {

template <bool Saturate>
int run(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b) {
  if (mode >> 32)
    return goc::execute_dpp(flags, exec_mask, mode, a,
                            [&](uint32_t exec_mask, const uint32_t *const *source) {
                              return run<Saturate>(flags, exec_mask, uint32_t(mode), d, source, b);
                            });
  const uint32_t known = Saturate ? GOC_ALU_HIGH_D | GOC_ALU_CLAMP
                                  : GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_ABS_A |
                                        GOC_ALU_ABS_B | GOC_ALU_NEG_A | GOC_ALU_NEG_B;
  if (int error = goc::validate(flags, mode & ~known, true,
                                Saturate ? 0
                                         : GOC_FP_IEEE_MODE | GOC_FP_FLUSH_INPUT_DENORMALS |
                                               GOC_FP_FLUSH_OUTPUT_DENORMALS))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
  const uint32_t *bp = nullptr;
  if constexpr (!Saturate)
    bp = b[0];
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::pack_x86_64_v4<Saturate>(exec_mask, mode, d[0], a[0], bp);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::pack_x86_64_v3<Saturate>(exec_mask, mode, d[0], a[0], bp);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (unsigned lane = 0; lane < 32; ++lane) {
    uint32_t x = a[0][lane];
    if constexpr (Saturate) {
      auto convert = [](uint32_t raw) {
        int32_t value = int32_t(raw) - ((raw & 32768) ? 65536 : 0);
        return uint32_t(std::clamp(value, 0, 255));
      };
      uint32_t packed = convert(x & 65535) | (convert(x >> 16) << 8);
      unsigned shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
      result[lane] = (d[0][lane] & ~(65535u << shift)) | (packed << shift);
    } else {
      x = (x >> (mode & GOC_ALU_HIGH_A ? 16 : 0)) & 65535;
      uint32_t y = (bp[lane] >> (mode & GOC_ALU_HIGH_B ? 16 : 0)) & 65535;
      if ((x & 32767) > 0x7c00)
        x |= 0x200;
      if ((y & 32767) > 0x7c00)
        y |= 0x200;
      if (mode & GOC_ALU_ABS_A)
        x &= 32767;
      if (mode & GOC_ALU_ABS_B)
        y &= 32767;
      if (mode & GOC_ALU_NEG_A)
        x ^= 32768;
      if (mode & GOC_ALU_NEG_B)
        y ^= 32768;
      result[lane] = x | (y << 16);
    }
  }
  for (unsigned lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

template <bool Saturate>
int run_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b) {
  if (int error = run<Saturate>(flags, 0, mode, nullptr, nullptr, nullptr))
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
    const uint32_t *source_a = a[0] + half * 32, *source_b = nullptr;
    if constexpr (!Saturate)
      source_b = b[0] + half * 32;
    if (int error = run<Saturate>(flags, uint32_t(exec_mask >> (half * 32)), half_mode, &dest,
                                  &source_a, &source_b))
      return error;
  }
  return GOC_SUCCESS;
}

} // namespace

int goc_v_sat_pk_u8_i16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a) {
  return run<true>(flags, exec_mask, instruction_flags, d, a, nullptr);
}

int goc_v_pack_b32_f16_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run<false>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_v_sat_pk_u8_i16_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a) {
  return run_wave64<true>(flags, exec_mask, instruction_flags, d, a, nullptr);
}

int goc_v_pack_b32_f16_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run_wave64<false>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_v_pack_b32_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (int error = goc::validate(flags, 0, false,
                                GOC_FP_IEEE_MODE | GOC_FP_FLUSH_INPUT_DENORMALS |
                                    GOC_FP_FLUSH_OUTPUT_DENORMALS))
    return error;
  return goc_v_pack_b32_f16_rdna4(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

int goc_v_pack_b32_f16_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (int error = goc::validate(flags, 0, false,
                                GOC_FP_IEEE_MODE | GOC_FP_FLUSH_INPUT_DENORMALS |
                                    GOC_FP_FLUSH_OUTPUT_DENORMALS))
    return error;
  return goc_v_pack_b32_f16_rdna4_wave64(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

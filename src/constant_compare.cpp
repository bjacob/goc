// SPDX-License-Identifier: MIT

// RDNA3 constant predicates follow rocjitsu execute_v_cmp_[ft]_* for results.
// Floating exception reporting needs a separate RDNA3 model.

#include "dpp.h"
#include "goc/goc.h"
#include "internal.h"

#include <stdint.h>

namespace {

template <typename Mask>
int compare(uint64_t flags, Mask exec_mask, uint64_t mode, Mask *d, bool value, unsigned bits,
            bool floating = false, uint32_t *trapsts = nullptr) {
  if (floating && trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32) {
    if (bits == 64 || !(mode & (GOC_DPP8 | GOC_DPP16)) || !goc::valid_dpp(mode))
      return GOC_ERROR_INVALID_FLAGS;
  }
  const uint32_t known = floating ? GOC_ALU_CLAMP | GOC_ALU_ABS_A | GOC_ALU_ABS_B | GOC_ALU_NEG_A |
                                        GOC_ALU_NEG_B |
                                        (bits == 16 ? GOC_ALU_HIGH_A | GOC_ALU_HIGH_B : 0)
                                  : 0;
  const uint64_t fp =
      floating ? GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS | GOC_FP_IEEE_MODE
               : 0;
  if (int error = goc::validate(flags, uint32_t(mode) & ~known, true, fp))
    return error;
  if (value && exec_mask && (mode & GOC_DPP16)) {
    // Operand values are irrelevant, but DPP row/bank/boundary filtering is not.
    uint32_t zero[32] = {}, ignored[32];
    Mask filtered = 0;
    const unsigned rows = (mode & GOC_DPP_ROW_MASK) >> GOC_DPP_ROW_SHIFT;
    for (unsigned half = 0; half < sizeof(Mask) / sizeof(uint32_t); ++half) {
      // DPP routes within 16-lane rows; Wave64 has four independent row bits.
      const uint64_t half_mode =
          (mode & ~GOC_DPP_ROW_MASK) | (uint64_t((rows >> (half * 2)) & 3) << GOC_DPP_ROW_SHIFT);
      filtered |= Mask(goc::dpp16_source(flags, uint32_t(exec_mask >> (half * 32)), half_mode,
                                         ignored, zero))
                  << (half * 32);
    }
    exec_mask = filtered;
  }
  *d = value ? exec_mask : 0;
  return GOC_SUCCESS;
}

} // namespace

int goc_v_cmp_f_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, false, 32);
}

int goc_v_cmp_t_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, true, 32);
}

int goc_v_cmpx_f_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, false, 32);
}

int goc_v_cmpx_t_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, true, 32);
}

int goc_v_cmp_f_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, false, 32);
}

int goc_v_cmp_t_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, true, 32);
}

int goc_v_cmpx_f_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, false, 32);
}

int goc_v_cmpx_t_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, true, 32);
}

int goc_v_cmp_f_i64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, false, 64);
}

int goc_v_cmp_t_i64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, true, 64);
}

int goc_v_cmpx_f_i64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, false, 64);
}

int goc_v_cmpx_t_i64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, true, 64);
}

int goc_v_cmp_f_u64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, false, 64);
}

int goc_v_cmp_t_u64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, true, 64);
}

int goc_v_cmpx_f_u64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, false, 64);
}

int goc_v_cmpx_t_u64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, true, 64);
}

int goc_v_cmp_f_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *, const uint32_t *const *, uint32_t *trapsts) {
  return compare(flags, exec_mask, instruction_flags, d, false, 16, true, trapsts);
}

int goc_v_cmp_t_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *, const uint32_t *const *, uint32_t *trapsts) {
  return compare(flags, exec_mask, instruction_flags, d, true, 16, true, trapsts);
}

int goc_v_cmpx_f_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *, const uint32_t *const *, uint32_t *trapsts) {
  return compare(flags, exec_mask, instruction_flags, d, false, 16, true, trapsts);
}

int goc_v_cmpx_t_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *, const uint32_t *const *, uint32_t *trapsts) {
  return compare(flags, exec_mask, instruction_flags, d, true, 16, true, trapsts);
}

int goc_v_cmp_f_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *, const uint32_t *const *, uint32_t *trapsts) {
  return compare(flags, exec_mask, instruction_flags, d, false, 32, true, trapsts);
}

int goc_v_cmp_t_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *, const uint32_t *const *, uint32_t *trapsts) {
  return compare(flags, exec_mask, instruction_flags, d, true, 32, true, trapsts);
}

int goc_v_cmpx_f_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *, const uint32_t *const *, uint32_t *trapsts) {
  return compare(flags, exec_mask, instruction_flags, d, false, 32, true, trapsts);
}

int goc_v_cmpx_t_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *, const uint32_t *const *, uint32_t *trapsts) {
  return compare(flags, exec_mask, instruction_flags, d, true, 32, true, trapsts);
}

int goc_v_cmp_f_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *, const uint32_t *const *, uint32_t *trapsts) {
  return compare(flags, exec_mask, instruction_flags, d, false, 64, true, trapsts);
}

int goc_v_cmp_t_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                    const uint32_t *const *, const uint32_t *const *, uint32_t *trapsts) {
  return compare(flags, exec_mask, instruction_flags, d, true, 64, true, trapsts);
}

int goc_v_cmpx_f_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *, const uint32_t *const *, uint32_t *trapsts) {
  return compare(flags, exec_mask, instruction_flags, d, false, 64, true, trapsts);
}

int goc_v_cmpx_t_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                     const uint32_t *const *, const uint32_t *const *, uint32_t *trapsts) {
  return compare(flags, exec_mask, instruction_flags, d, true, 64, true, trapsts);
}

int goc_v_cmp_f_i32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint64_t *d, const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, false, 32);
}

int goc_v_cmp_t_i32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint64_t *d, const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, true, 32);
}

int goc_v_cmpx_f_i32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint64_t *d, const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, false, 32);
}

int goc_v_cmpx_t_i32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint64_t *d, const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, true, 32);
}

int goc_v_cmp_f_u32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint64_t *d, const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, false, 32);
}

int goc_v_cmp_t_u32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint64_t *d, const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, true, 32);
}

int goc_v_cmpx_f_u32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint64_t *d, const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, false, 32);
}

int goc_v_cmpx_t_u32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint64_t *d, const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, true, 32);
}

int goc_v_cmp_f_i64_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint64_t *d, const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, false, 64);
}

int goc_v_cmp_t_i64_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint64_t *d, const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, true, 64);
}

int goc_v_cmpx_f_i64_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint64_t *d, const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, false, 64);
}

int goc_v_cmpx_t_i64_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint64_t *d, const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, true, 64);
}

int goc_v_cmp_f_u64_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint64_t *d, const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, false, 64);
}

int goc_v_cmp_t_u64_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint64_t *d, const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, true, 64);
}

int goc_v_cmpx_f_u64_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint64_t *d, const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, false, 64);
}

int goc_v_cmpx_t_u64_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint64_t *d, const uint32_t *const *, const uint32_t *const *) {
  return compare(flags, exec_mask, instruction_flags, d, true, 64);
}

int goc_v_cmp_f_f16_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint64_t *d, const uint32_t *const *, const uint32_t *const *,
                           uint32_t *trapsts) {
  return compare(flags, exec_mask, instruction_flags, d, false, 16, true, trapsts);
}

int goc_v_cmp_t_f16_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint64_t *d, const uint32_t *const *, const uint32_t *const *,
                           uint32_t *trapsts) {
  return compare(flags, exec_mask, instruction_flags, d, true, 16, true, trapsts);
}

int goc_v_cmpx_f_f16_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint64_t *d, const uint32_t *const *, const uint32_t *const *,
                            uint32_t *trapsts) {
  return compare(flags, exec_mask, instruction_flags, d, false, 16, true, trapsts);
}

int goc_v_cmpx_t_f16_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint64_t *d, const uint32_t *const *, const uint32_t *const *,
                            uint32_t *trapsts) {
  return compare(flags, exec_mask, instruction_flags, d, true, 16, true, trapsts);
}

int goc_v_cmp_f_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint64_t *d, const uint32_t *const *, const uint32_t *const *,
                           uint32_t *trapsts) {
  return compare(flags, exec_mask, instruction_flags, d, false, 32, true, trapsts);
}

int goc_v_cmp_t_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint64_t *d, const uint32_t *const *, const uint32_t *const *,
                           uint32_t *trapsts) {
  return compare(flags, exec_mask, instruction_flags, d, true, 32, true, trapsts);
}

int goc_v_cmpx_f_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint64_t *d, const uint32_t *const *, const uint32_t *const *,
                            uint32_t *trapsts) {
  return compare(flags, exec_mask, instruction_flags, d, false, 32, true, trapsts);
}

int goc_v_cmpx_t_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint64_t *d, const uint32_t *const *, const uint32_t *const *,
                            uint32_t *trapsts) {
  return compare(flags, exec_mask, instruction_flags, d, true, 32, true, trapsts);
}

int goc_v_cmp_f_f64_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint64_t *d, const uint32_t *const *, const uint32_t *const *,
                           uint32_t *trapsts) {
  return compare(flags, exec_mask, instruction_flags, d, false, 64, true, trapsts);
}

int goc_v_cmp_t_f64_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint64_t *d, const uint32_t *const *, const uint32_t *const *,
                           uint32_t *trapsts) {
  return compare(flags, exec_mask, instruction_flags, d, true, 64, true, trapsts);
}

int goc_v_cmpx_f_f64_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint64_t *d, const uint32_t *const *, const uint32_t *const *,
                            uint32_t *trapsts) {
  return compare(flags, exec_mask, instruction_flags, d, false, 64, true, trapsts);
}

int goc_v_cmpx_t_f64_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint64_t *d, const uint32_t *const *, const uint32_t *const *,
                            uint32_t *trapsts) {
  return compare(flags, exec_mask, instruction_flags, d, true, 64, true, trapsts);
}

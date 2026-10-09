// SPDX-License-Identifier: MIT

#include "half_binary.h"
#include "alu.h"
#include "binary.h"
#include "dpp.h"
#include "float_formats.h"
#include "goc/goc.h"
#include "internal.h"
#include "minmax.h"
#include "packed_alu.h"

#include <stdint.h>

namespace {

template <goc::Binary Op, bool Packed = false>
int binary(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
           const uint32_t *const *a, const uint32_t *const *b) {
  if (mode >> 32) {
    if constexpr (!Packed) {
      return goc::execute_dpp(
          flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
            return binary<Op, Packed>(flags, exec_mask, uint32_t(mode), d, source, b);
          });
    } else {
      return GOC_ERROR_INVALID_FLAGS;
    }
  }

  const uint32_t known = Packed ? GOC_PK_NEG_LO_A | GOC_PK_NEG_LO_B | GOC_PK_NEG_HI_A |
                                      GOC_PK_NEG_HI_B | GOC_PK_CLAMP | GOC_PK_LO_A_HIGH |
                                      GOC_PK_LO_B_HIGH | GOC_PK_HI_A_LOW | GOC_PK_HI_B_LOW
                                : GOC_ALU_NEG_A | GOC_ALU_NEG_B | GOC_ALU_ABS_A | GOC_ALU_ABS_B |
                                      GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP | GOC_ALU_HIGH_A |
                                      GOC_ALU_HIGH_B | GOC_ALU_HIGH_D;
  if (int error = goc::validate(flags, mode & ~known))
    return error;
  if (exec_mask == 0)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::half_binary_x86_64_v3<Op, Packed>(bool(flags & GOC_FP16_OVFL), exec_mask, mode, d[0], a[0],
                                           b[0]);
    return GOC_SUCCESS;
  }
#endif
  uint32_t modes[] = {Packed ? goc::packed_half_mode(mode, false) : uint32_t(mode),
                      goc::packed_half_mode(mode, true)};
  int d_shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    result[lane] = 0;
    for (int half = 0; half < (Packed ? 2 : 1); ++half) {
      uint32_t m = modes[half];
      int a_shift = m & GOC_ALU_HIGH_A ? 16 : 0;
      int b_shift = m & GOC_ALU_HIGH_B ? 16 : 0;
      float x = goc::alu_input(goc::as_bits(goc::f16_to_float(uint16_t(a[0][lane] >> a_shift))), m);
      float y =
          goc::alu_input(goc::as_bits(goc::f16_to_float(uint16_t(b[0][lane] >> b_shift))), m >> 1);
      float value;
      if constexpr (Op == goc::Binary::Add)
        value = x + y;
      if constexpr (Op == goc::Binary::Sub)
        value = x - y;
      if constexpr (Op == goc::Binary::Subrev)
        value = y - x;
      if constexpr (Op == goc::Binary::Mul)
        value = x * y;
      if constexpr (Op == goc::Binary::MinNum)
        value = goc::minmax<false, false>(x, y);
      if constexpr (Op == goc::Binary::MaxNum)
        value = goc::minmax<true, false>(x, y);
      if constexpr (Op == goc::Binary::Minimum)
        value = goc::minmax<false, true>(x, y);
      if constexpr (Op == goc::Binary::Maximum)
        value = goc::minmax<true, true>(x, y);
      result[lane] |=
          uint32_t(goc::float_to_f16(goc::alu_output_f16(value, m), flags & GOC_FP16_OVFL))
          << (16 * half);
    }
  }
  for (int lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1) {
      if constexpr (Packed)
        d[0][lane] = result[lane];
      else
        d[0][lane] = (d[0][lane] & ~(0xffffU << d_shift)) | (result[lane] << d_shift);
    }
  return GOC_SUCCESS;
}

} // namespace

int goc_v_add_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary<goc::Binary::Add>(flags, exec_mask, mode, d, a, b);
}

int goc_v_sub_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary<goc::Binary::Sub>(flags, exec_mask, mode, d, a, b);
}

int goc_v_subrev_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary<goc::Binary::Subrev>(flags, exec_mask, mode, d, a, b);
}

int goc_v_mul_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary<goc::Binary::Mul>(flags, exec_mask, mode, d, a, b);
}

int goc_v_min_num_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary<goc::Binary::MinNum>(flags, exec_mask, mode, d, a, b);
}

int goc_v_max_num_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary<goc::Binary::MaxNum>(flags, exec_mask, mode, d, a, b);
}

int goc_v_minimum_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary<goc::Binary::Minimum>(flags, exec_mask, mode, d, a, b);
}

int goc_v_maximum_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary<goc::Binary::Maximum>(flags, exec_mask, mode, d, a, b);
}

int goc_v_pk_add_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return binary<goc::Binary::Add, true>(flags, exec_mask, mode, d, a, b);
}

int goc_v_pk_mul_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return binary<goc::Binary::Mul, true>(flags, exec_mask, mode, d, a, b);
}

int goc_v_pk_min_num_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b,
                         uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return binary<goc::Binary::MinNum, true>(flags, exec_mask, mode, d, a, b);
}

int goc_v_pk_max_num_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b,
                         uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return binary<goc::Binary::MaxNum, true>(flags, exec_mask, mode, d, a, b);
}

int goc_v_pk_minimum_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b,
                         uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return binary<goc::Binary::Minimum, true>(flags, exec_mask, mode, d, a, b);
}

int goc_v_pk_maximum_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b,
                         uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return binary<goc::Binary::Maximum, true>(flags, exec_mask, mode, d, a, b);
}

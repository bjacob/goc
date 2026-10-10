// SPDX-License-Identifier: MIT

#include "binary.h"
#include "alu.h"
#include "dpp.h"
#include "goc/goc.h"
#include "internal.h"
#include "minmax.h"

#include <stdint.h>

namespace {

template <goc::Binary Op, bool Rdna3 = false>
int binary(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
           const uint32_t *const *a, const uint32_t *const *b) {
  if (mode >> 32) {
    return goc::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return binary<Op, Rdna3>(flags, exec_mask, uint32_t(mode), d, source, b);
        });
  }

  const uint32_t known = GOC_ALU_NEG_A | GOC_ALU_NEG_B | GOC_ALU_ABS_A | GOC_ALU_ABS_B |
                         GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP;
  constexpr uint64_t fp_flags =
      GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS | GOC_FP_IEEE_MODE;
  if (int error = goc::validate(flags, mode & ~known, false, fp_flags))
    return error;
  // RDNA3 ignores output scaling when denormal outputs are preserved or IEEE
  // mode is enabled. RDNA4 applies it independently of either guest setting.
  if constexpr (Rdna3)
    if (!(flags & GOC_FP_FLUSH_OUTPUT_DENORMALS) || (flags & GOC_FP_IEEE_MODE))
      mode &= ~uint64_t(GOC_ALU_OMOD_HALF);
  if (exec_mask == 0)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::binary_x86_64_v3(Op, flags, exec_mask, mode, d[0], a[0], b[0]);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    float x = goc::alu_input(a[0][lane], mode), y = goc::alu_input(b[0][lane], mode >> 1);
    if (flags & GOC_FP_FLUSH_INPUT_DENORMALS) {
      x = goc::flush_denorm_f32(x);
      y = goc::flush_denorm_f32(y);
    }
    float value;
    if constexpr (Op == goc::Binary::Add)
      value = x + y;
    if constexpr (Op == goc::Binary::Sub)
      value = x - y;
    if constexpr (Op == goc::Binary::Subrev)
      value = y - x;
    if constexpr (Op == goc::Binary::MulDx9Zero)
      value = (x == 0 || y == 0) ? 0.0f : x * y;
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
    if constexpr (Op == goc::Binary::MinLegacy || Op == goc::Binary::MaxLegacy)
      value = goc::minmax_rdna3<Op == goc::Binary::MaxLegacy>(x, y, flags & GOC_FP_IEEE_MODE);
    value = goc::alu_output_f32(value, mode);
    if constexpr (Op == goc::Binary::Add || Op == goc::Binary::Sub || Op == goc::Binary::Subrev ||
                  Op == goc::Binary::Mul || Op == goc::Binary::MulDx9Zero)
      if (flags & GOC_FP_FLUSH_OUTPUT_DENORMALS)
        value = goc::flush_denorm_f32(value);
    result[lane] = goc::as_bits(value);
  }
  for (int lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

template <goc::Binary Op, bool Rdna3 = false>
int binary_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, const uint32_t *const *b) {
  if (int error = binary<Op, Rdna3>(flags, 0, mode, nullptr, nullptr, nullptr))
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
    const uint32_t *x = a[0] + half * 32, *y = b[0] + half * 32;
    if (int error =
            binary<Op, Rdna3>(flags, uint32_t(exec_mask >> (half * 32)), half_mode, &dest, &x, &y))
      return error;
  }
  return GOC_SUCCESS;
}

} // namespace

int goc_v_add_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary<goc::Binary::Add, true>(flags, exec_mask, mode, d, a, b);
}

int goc_v_add_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary<goc::Binary::Add>(flags, exec_mask, mode, d, a, b);
}

int goc_v_sub_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary<goc::Binary::Sub, true>(flags, exec_mask, mode, d, a, b);
}

int goc_v_sub_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary<goc::Binary::Sub>(flags, exec_mask, mode, d, a, b);
}

int goc_v_subrev_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary<goc::Binary::Subrev, true>(flags, exec_mask, mode, d, a, b);
}

int goc_v_subrev_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b,
                           uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary<goc::Binary::Subrev>(flags, exec_mask, mode, d, a, b);
}

int goc_v_mul_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary<goc::Binary::Mul, true>(flags, exec_mask, mode, d, a, b);
}

int goc_v_mul_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary<goc::Binary::Mul>(flags, exec_mask, mode, d, a, b);
}

int goc_v_min_num_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary<goc::Binary::MinNum>(flags, exec_mask, mode, d, a, b);
}

int goc_v_max_num_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary<goc::Binary::MaxNum>(flags, exec_mask, mode, d, a, b);
}

int goc_v_minimum_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary<goc::Binary::Minimum>(flags, exec_mask, mode, d, a, b);
}

int goc_v_maximum_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary<goc::Binary::Maximum>(flags, exec_mask, mode, d, a, b);
}

int goc_v_mul_dx9_zero_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary<goc::Binary::MulDx9Zero, true>(flags, exec_mask, mode, d, a, b);
}

int goc_v_mul_dx9_zero_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary<goc::Binary::MulDx9Zero>(flags, exec_mask, mode, d, a, b);
}

int goc_v_min_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary<goc::Binary::MinLegacy, true>(flags, exec_mask, mode, d, a, b);
}

int goc_v_max_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                  const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary<goc::Binary::MaxLegacy, true>(flags, exec_mask, mode, d, a, b);
}

int goc_v_add_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary_wave64<goc::Binary::Add, true>(flags, exec_mask, mode, d, a, b);
}

int goc_v_add_f32_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary_wave64<goc::Binary::Add>(flags, exec_mask, mode, d, a, b);
}

int goc_v_sub_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary_wave64<goc::Binary::Sub, true>(flags, exec_mask, mode, d, a, b);
}

int goc_v_sub_f32_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary_wave64<goc::Binary::Sub>(flags, exec_mask, mode, d, a, b);
}

int goc_v_subrev_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary_wave64<goc::Binary::Subrev, true>(flags, exec_mask, mode, d, a, b);
}

int goc_v_subrev_f32_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary_wave64<goc::Binary::Subrev>(flags, exec_mask, mode, d, a, b);
}

int goc_v_mul_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary_wave64<goc::Binary::Mul, true>(flags, exec_mask, mode, d, a, b);
}

int goc_v_mul_f32_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary_wave64<goc::Binary::Mul>(flags, exec_mask, mode, d, a, b);
}

int goc_v_min_num_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary_wave64<goc::Binary::MinNum>(flags, exec_mask, mode, d, a, b);
}

int goc_v_max_num_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary_wave64<goc::Binary::MaxNum>(flags, exec_mask, mode, d, a, b);
}

int goc_v_minimum_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary_wave64<goc::Binary::Minimum>(flags, exec_mask, mode, d, a, b);
}

int goc_v_maximum_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary_wave64<goc::Binary::Maximum>(flags, exec_mask, mode, d, a, b);
}

int goc_v_mul_dx9_zero_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary_wave64<goc::Binary::MulDx9Zero, true>(flags, exec_mask, mode, d, a, b);
}

int goc_v_mul_dx9_zero_f32_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                        uint32_t *const *d, const uint32_t *const *a,
                                        const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary_wave64<goc::Binary::MulDx9Zero>(flags, exec_mask, mode, d, a, b);
}

int goc_v_min_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary_wave64<goc::Binary::MinLegacy, true>(flags, exec_mask, mode, d, a, b);
}

int goc_v_max_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b, uint32_t *trapsts) {
  if (trapsts && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return binary_wave64<goc::Binary::MaxLegacy, true>(flags, exec_mask, mode, d, a, b);
}

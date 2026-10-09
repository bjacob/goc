// SPDX-License-Identifier: MIT

#include "rdna4_binary.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_alu.h"
#include "rdna4_dpp.h"
#include "rdna4_minmax.h"

#include <stdint.h>

namespace {

template <goc::Binary Op>
int binary(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
           const uint32_t *const *a, const uint32_t *const *b) {
  if (mode >> 32) {
    return goc::execute_dpp(flags, exec_mask, mode, a,
                            [&](uint32_t exec_mask, const uint32_t *const *source) {
                              return binary<Op>(flags, exec_mask, uint32_t(mode), d, source, b);
                            });
  }

  const uint32_t known = GOC_ALU_NEG_A | GOC_ALU_NEG_B | GOC_ALU_ABS_A | GOC_ALU_ABS_B |
                         GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP;
  if (int error = goc::validate(flags, mode & ~known))
    return error;
  if (exec_mask == 0)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::binary_x86_64_v3(Op, exec_mask, mode, d[0], a[0], b[0]);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    float x = goc::alu_input(a[0][lane], mode), y = goc::alu_input(b[0][lane], mode >> 1);
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
    result[lane] = goc::as_bits(goc::alu_output_f32(value, mode));
  }
  for (int lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_add_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        uint32_t *excp_flag_user) {
  if (excp_flag_user)
    return GOC_ERROR_UNSUPPORTED_EXCEPTIONS;
  return binary<goc::Binary::Add>(flags, exec_mask, mode, d, a, b);
}

int goc_rdna4_v_sub_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        uint32_t *excp_flag_user) {
  if (excp_flag_user)
    return GOC_ERROR_UNSUPPORTED_EXCEPTIONS;
  return binary<goc::Binary::Sub>(flags, exec_mask, mode, d, a, b);
}

int goc_rdna4_v_subrev_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b,
                           uint32_t *excp_flag_user) {
  if (excp_flag_user)
    return GOC_ERROR_UNSUPPORTED_EXCEPTIONS;
  return binary<goc::Binary::Subrev>(flags, exec_mask, mode, d, a, b);
}

int goc_rdna4_v_mul_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        uint32_t *excp_flag_user) {
  if (excp_flag_user)
    return GOC_ERROR_UNSUPPORTED_EXCEPTIONS;
  return binary<goc::Binary::Mul>(flags, exec_mask, mode, d, a, b);
}

int goc_rdna4_v_min_num_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            uint32_t *excp_flag_user) {
  if (excp_flag_user)
    return GOC_ERROR_UNSUPPORTED_EXCEPTIONS;
  return binary<goc::Binary::MinNum>(flags, exec_mask, mode, d, a, b);
}

int goc_rdna4_v_max_num_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            uint32_t *excp_flag_user) {
  if (excp_flag_user)
    return GOC_ERROR_UNSUPPORTED_EXCEPTIONS;
  return binary<goc::Binary::MaxNum>(flags, exec_mask, mode, d, a, b);
}

int goc_rdna4_v_minimum_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            uint32_t *excp_flag_user) {
  if (excp_flag_user)
    return GOC_ERROR_UNSUPPORTED_EXCEPTIONS;
  return binary<goc::Binary::Minimum>(flags, exec_mask, mode, d, a, b);
}

int goc_rdna4_v_maximum_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            uint32_t *excp_flag_user) {
  if (excp_flag_user)
    return GOC_ERROR_UNSUPPORTED_EXCEPTIONS;
  return binary<goc::Binary::Maximum>(flags, exec_mask, mode, d, a, b);
}

int goc_rdna4_v_mul_dx9_zero_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user)
    return GOC_ERROR_UNSUPPORTED_EXCEPTIONS;
  return binary<goc::Binary::MulDx9Zero>(flags, exec_mask, mode, d, a, b);
}

// SPDX-License-Identifier: MIT

// Integer models adapted from rocjitsu generated/shared/execute_shared.h.
// GFX1201 captures correct its ABSDIFF model: subtract wraps before ABS.

#include "goc/goc.h"
#include "internal.h"
#include "rdna4_bits.h"

#include <stdint.h>

namespace {

int validate(uint64_t flags, uint64_t mode) {
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return goc::validate(flags, uint32_t(mode), true);
}

template <bool Subtract, bool Signed, bool CarryIn = false>
int carry(uint64_t flags, uint64_t mode, uint32_t *d, uint32_t a, uint32_t b, uint32_t *scc,
          uint32_t input_scc = 0) {
  if (int error = validate(flags, mode))
    return error;
  uint32_t result, cc;
  if constexpr (Signed) {
    result = Subtract ? a - b : a + b;
    cc = ((Subtract ? a ^ b : ~(a ^ b)) & (a ^ result)) >> 31;
  } else {
    uint64_t right = uint64_t(b) + (CarryIn ? input_scc & 1 : 0);
    uint64_t wide = Subtract ? uint64_t(a) - right : uint64_t(a) + right;
    result = uint32_t(wide);
    cc = Subtract ? uint64_t(a) < right : wide >> 32;
  }
  *d = result;
  *scc = cc;
  return GOC_SUCCESS;
}

template <bool Difference>
int absolute(uint64_t flags, uint64_t mode, uint32_t *d, uint32_t a, uint32_t b, uint32_t *scc) {
  if (int error = validate(flags, mode))
    return error;
  uint32_t value = Difference ? a - b : a;
  uint32_t result = value & 0x80000000u ? 0u - value : value, cc = result != 0;
  *d = result;
  *scc = cc;
  return GOC_SUCCESS;
}

template <bool Maximum, bool Signed>
int minmax(uint64_t flags, uint64_t mode, uint32_t *d, uint32_t a, uint32_t b, uint32_t *scc) {
  if (int error = validate(flags, mode))
    return error;
  uint32_t x = a ^ (Signed ? 0x80000000u : 0), y = b ^ (Signed ? 0x80000000u : 0);
  uint32_t cc = Maximum ? x > y : x < y, result = cc ? a : b;
  *d = result;
  *scc = cc;
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_s_add_co_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  return carry<false, false>(flags, instruction_flags, d, a, b, scc);
}

int goc_rdna4_s_sub_co_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  return carry<true, false>(flags, instruction_flags, d, a, b, scc);
}

int goc_rdna4_s_add_co_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  return carry<false, true>(flags, instruction_flags, d, a, b, scc);
}

int goc_rdna4_s_sub_co_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  return carry<true, true>(flags, instruction_flags, d, a, b, scc);
}

int goc_rdna4_s_add_co_ci_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, uint32_t a, uint32_t b, uint32_t *scc,
                              uint32_t input_scc) {
  (void)exec_mask;
  return carry<false, false, true>(flags, instruction_flags, d, a, b, scc, input_scc);
}

int goc_rdna4_s_sub_co_ci_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, uint32_t a, uint32_t b, uint32_t *scc,
                              uint32_t input_scc) {
  (void)exec_mask;
  return carry<true, false, true>(flags, instruction_flags, d, a, b, scc, input_scc);
}

int goc_rdna4_s_abs_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                        uint32_t a, uint32_t *scc) {
  (void)exec_mask;
  return absolute<false>(flags, instruction_flags, d, a, 0, scc);
}

int goc_rdna4_s_absdiff_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  return absolute<true>(flags, instruction_flags, d, a, b, scc);
}

int goc_rdna4_s_min_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                        uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  return minmax<false, true>(flags, instruction_flags, d, a, b, scc);
}

int goc_rdna4_s_min_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                        uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  return minmax<false, false>(flags, instruction_flags, d, a, b, scc);
}

int goc_rdna4_s_max_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                        uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  return minmax<true, true>(flags, instruction_flags, d, a, b, scc);
}

int goc_rdna4_s_max_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                        uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  return minmax<true, false>(flags, instruction_flags, d, a, b, scc);
}

int goc_rdna4_s_mul_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                        uint32_t a, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = a * b;
  *d = result;
  return GOC_SUCCESS;
}

int goc_rdna4_s_mul_hi_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, uint32_t a, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = (uint64_t(a) * b) >> 32;
  *d = result;
  return GOC_SUCCESS;
}

int goc_rdna4_s_mul_hi_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *d, uint32_t a, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  int64_t x = goc::extend_integer<32, true>(a);
  int64_t y = goc::extend_integer<32, true>(b);
  uint32_t result = uint64_t(x * y) >> 32;
  *d = result;
  return GOC_SUCCESS;
}

int goc_rdna4_s_add_nc_u64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint64_t *d, uint64_t a, uint64_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t result = a + b;
  *d = result;
  return GOC_SUCCESS;
}

int goc_rdna4_s_sub_nc_u64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint64_t *d, uint64_t a, uint64_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t result = a - b;
  *d = result;
  return GOC_SUCCESS;
}

int goc_rdna4_s_mul_u64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint64_t *d,
                        uint64_t a, uint64_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t result = a * b;
  *d = result;
  return GOC_SUCCESS;
}

int goc_rdna4_s_addk_co_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, uint16_t immediate, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t a = *d, b = uint32_t(immediate) | ((immediate & 0x8000) ? 0xffff0000u : 0);
  uint32_t result = a + b, cc = (~(a ^ b) & (a ^ result)) >> 31;
  *d = result;
  *scc = cc;
  return GOC_SUCCESS;
}

int goc_rdna4_s_mulk_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *d, uint16_t immediate) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t b = uint32_t(immediate) | ((immediate & 0x8000) ? 0xffff0000u : 0);
  uint32_t result = *d * b;
  *d = result;
  return GOC_SUCCESS;
}

int goc_rdna4_s_sext_i32_i8(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, uint32_t a) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  *d = goc::sign_extend_word<8>(a);
  return GOC_SUCCESS;
}

int goc_rdna4_s_sext_i32_i16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, uint32_t a) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  *d = goc::sign_extend_word<16>(a);
  return GOC_SUCCESS;
}

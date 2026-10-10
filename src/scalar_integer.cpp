// SPDX-License-Identifier: MIT

// Integer models adapted from rocjitsu generated/shared/execute_shared.h.
// GFX1201 captures correct its ABSDIFF model: subtract wraps before ABS.

#include "bits.h"
#include "goc/goc.h"
#include "internal.h"

#include <stdint.h>

int goc_s_add_co_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                     uint32_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t wide = uint64_t(a) + b;
  uint32_t result = uint32_t(wide), cc = wide >> 32;
  *d = result;
  *scc = cc;
  return GOC_SUCCESS;
}

int goc_s_sub_co_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                     uint32_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = a - b, cc = a < b;
  *d = result;
  *scc = cc;
  return GOC_SUCCESS;
}

int goc_s_add_co_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                     uint32_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = a + b, cc = (~(a ^ b) & (a ^ result)) >> 31;
  *d = result;
  *scc = cc;
  return GOC_SUCCESS;
}

int goc_s_sub_co_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                     uint32_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = a - b, cc = ((a ^ b) & (a ^ result)) >> 31;
  *d = result;
  *scc = cc;
  return GOC_SUCCESS;
}

int goc_s_add_co_ci_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t b, uint32_t *scc, uint32_t input_scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t wide = uint64_t(a) + b + (input_scc & 1);
  uint32_t result = uint32_t(wide), cc = wide >> 32;
  *d = result;
  *scc = cc;
  return GOC_SUCCESS;
}

int goc_s_sub_co_ci_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t b, uint32_t *scc, uint32_t input_scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t subtrahend = uint64_t(b) + (input_scc & 1);
  uint32_t result = a - uint32_t(subtrahend), cc = uint64_t(a) < subtrahend;
  *d = result;
  *scc = cc;
  return GOC_SUCCESS;
}

int goc_s_abs_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                  uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = a & 0x80000000u ? 0u - a : a, cc = result != 0;
  *d = result;
  *scc = cc;
  return GOC_SUCCESS;
}

int goc_s_absdiff_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                      uint32_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t difference = a - b;
  uint32_t result = difference & 0x80000000u ? 0u - difference : difference, cc = result != 0;
  *d = result;
  *scc = cc;
  return GOC_SUCCESS;
}

int goc_s_min_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                  uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t cc = (a ^ 0x80000000u) < (b ^ 0x80000000u), result = cc ? a : b;
  *d = result;
  *scc = cc;
  return GOC_SUCCESS;
}

int goc_s_min_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                  uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t cc = a < b, result = cc ? a : b;
  *d = result;
  *scc = cc;
  return GOC_SUCCESS;
}

int goc_s_max_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                  uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t cc = (a ^ 0x80000000u) > (b ^ 0x80000000u), result = cc ? a : b;
  *d = result;
  *scc = cc;
  return GOC_SUCCESS;
}

int goc_s_max_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                  uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t cc = a > b, result = cc ? a : b;
  *d = result;
  *scc = cc;
  return GOC_SUCCESS;
}

int goc_s_mul_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = a * b;
  *d = result;
  return GOC_SUCCESS;
}

int goc_s_mul_hi_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                     uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = (uint64_t(a) * b) >> 32;
  *d = result;
  return GOC_SUCCESS;
}

int goc_s_mul_hi_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                     uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  int64_t x = goc::extend_integer<32, true>(a);
  int64_t y = goc::extend_integer<32, true>(b);
  uint32_t result = uint64_t(x * y) >> 32;
  *d = result;
  return GOC_SUCCESS;
}

int goc_s_add_nc_u64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                     uint64_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t result = a + b;
  *d = result;
  return GOC_SUCCESS;
}

int goc_s_sub_nc_u64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                     uint64_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t result = a - b;
  *d = result;
  return GOC_SUCCESS;
}

int goc_s_mul_u64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a, uint64_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t result = a * b;
  *d = result;
  return GOC_SUCCESS;
}

int goc_s_addk_co_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint16_t immediate,
                      uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t a = *d, b = uint32_t(immediate) | ((immediate & 0x8000) ? 0xffff0000u : 0);
  uint32_t result = a + b, cc = (~(a ^ b) & (a ^ result)) >> 31;
  *d = result;
  *scc = cc;
  return GOC_SUCCESS;
}

int goc_s_mulk_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint16_t immediate) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t b = uint32_t(immediate) | ((immediate & 0x8000) ? 0xffff0000u : 0);
  uint32_t result = *d * b;
  *d = result;
  return GOC_SUCCESS;
}

int goc_s_sext_i32_i8(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  *d = goc::sign_extend_word<8>(a);
  return GOC_SUCCESS;
}

int goc_s_sext_i32_i16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  *d = goc::sign_extend_word<16>(a);
  return GOC_SUCCESS;
}

// RDNA3 spellings share the RDNA4 carry/overflow implementations.
int goc_s_add_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                  uint32_t *scc) {
  return goc_s_add_co_u32(flags, instruction_flags, d, a, b, scc);
}

int goc_s_sub_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                  uint32_t *scc) {
  return goc_s_sub_co_u32(flags, instruction_flags, d, a, b, scc);
}

int goc_s_add_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                  uint32_t *scc) {
  return goc_s_add_co_i32(flags, instruction_flags, d, a, b, scc);
}

int goc_s_sub_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                  uint32_t *scc) {
  return goc_s_sub_co_i32(flags, instruction_flags, d, a, b, scc);
}

int goc_s_addc_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                   uint32_t input_scc, uint32_t *scc) {
  return goc_s_add_co_ci_u32(flags, instruction_flags, d, a, b, scc, input_scc);
}

int goc_s_subb_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a, uint32_t b,
                   uint32_t input_scc, uint32_t *scc) {
  return goc_s_sub_co_ci_u32(flags, instruction_flags, d, a, b, scc, input_scc);
}

int goc_s_addk_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint16_t immediate,
                   uint32_t *scc) {
  return goc_s_addk_co_i32(flags, instruction_flags, d, immediate, scc);
}

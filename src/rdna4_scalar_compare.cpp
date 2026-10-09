// SPDX-License-Identifier: MIT

// Raw float ordering follows rocjitsu shared/comparison.h and reuses the vector
// predicate helpers. Integer and bit tests follow generated/execute_shared.h.

#include "goc/goc.h"
#include "internal.h"
#include "rdna4_float_compare.h"
#include "rdna4_integer_compare.h"

#include <cstring>
#include <stdint.h>

namespace {

int validate(uint64_t flags, uint32_t mode) {
  return goc::validate(flags, mode, true,
                       GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS);
}

// Byte stores also allow SCC to overlap a word of a caller's uint64_t storage.
void store(uint32_t *scc, uint32_t result) { std::memcpy(scc, &result, sizeof(result)); }

template <bool Signed, unsigned Predicate, typename T>
int integer(uint64_t flags, uint32_t mode, uint32_t *scc, T a, T b) {
  if (int error = validate(flags, mode))
    return error;
  if constexpr (Signed) {
    a ^= T(1) << (sizeof(T) * 8 - 1);
    b ^= T(1) << (sizeof(T) * 8 - 1);
  }
  store(scc, goc::integer_compare_result<Predicate>(a < b, a == b) & 1);
  return GOC_SUCCESS;
}

template <unsigned BitValue, typename T>
int bit(uint64_t flags, uint32_t mode, uint32_t *scc, T a, uint32_t b) {
  if (int error = validate(flags, mode))
    return error;
  store(scc, ((a >> (b & (sizeof(T) * 8 - 1))) & 1) == BitValue);
  return GOC_SUCCESS;
}

template <unsigned Bits, unsigned Predicate>
int floating(uint64_t flags, uint32_t mode, uint32_t *scc, uint32_t a, uint32_t b,
             uint32_t *excp_flag_user) {
  if (int error = validate(flags, mode))
    return error;
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) == GOC_SEMANTICS_EXACT_EMPIRICAL)
    *excp_flag_user |=
        goc::float_compare_exceptions<Bits>(a, b, flags & GOC_FP_FLUSH_INPUT_DENORMALS);
  constexpr uint32_t sign = 1u << (Bits - 1), magnitude = sign - 1,
                     infinity = Bits == 16 ? 0x7c00 : 0x7f800000;
  if constexpr (Bits == 16) {
    a &= 65535;
    b &= 65535;
  }
  if (flags & GOC_FP_FLUSH_INPUT_DENORMALS) {
    if (!(a & infinity))
      a &= sign;
    if (!(b & infinity))
      b &= sign;
  }
  bool ordered = (a & magnitude) <= infinity && (b & magnitude) <= infinity;
  if (!(a & magnitude))
    a = 0;
  if (!(b & magnitude))
    b = 0;
  a ^= a & sign ? (sign | magnitude) : sign;
  b ^= b & sign ? (sign | magnitude) : sign;
  store(scc, goc::float_compare_result<Predicate>(a < b, a == b, ordered) & 1);
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_s_cmp_eq_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *scc, uint32_t a, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return integer<true, 2>(flags, instruction_flags, scc, a, b);
}

int goc_rdna4_s_cmp_lg_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *scc, uint32_t a, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return integer<true, 5>(flags, instruction_flags, scc, a, b);
}

int goc_rdna4_s_cmp_gt_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *scc, uint32_t a, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return integer<true, 4>(flags, instruction_flags, scc, a, b);
}

int goc_rdna4_s_cmp_ge_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *scc, uint32_t a, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return integer<true, 6>(flags, instruction_flags, scc, a, b);
}

int goc_rdna4_s_cmp_lt_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *scc, uint32_t a, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return integer<true, 1>(flags, instruction_flags, scc, a, b);
}

int goc_rdna4_s_cmp_le_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *scc, uint32_t a, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return integer<true, 3>(flags, instruction_flags, scc, a, b);
}

int goc_rdna4_s_cmp_eq_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *scc, uint32_t a, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return integer<false, 2>(flags, instruction_flags, scc, a, b);
}

int goc_rdna4_s_cmp_lg_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *scc, uint32_t a, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return integer<false, 5>(flags, instruction_flags, scc, a, b);
}

int goc_rdna4_s_cmp_gt_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *scc, uint32_t a, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return integer<false, 4>(flags, instruction_flags, scc, a, b);
}

int goc_rdna4_s_cmp_ge_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *scc, uint32_t a, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return integer<false, 6>(flags, instruction_flags, scc, a, b);
}

int goc_rdna4_s_cmp_lt_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *scc, uint32_t a, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return integer<false, 1>(flags, instruction_flags, scc, a, b);
}

int goc_rdna4_s_cmp_le_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *scc, uint32_t a, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return integer<false, 3>(flags, instruction_flags, scc, a, b);
}

int goc_rdna4_s_bitcmp0_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *scc, uint32_t a, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return bit<0>(flags, instruction_flags, scc, a, b);
}

int goc_rdna4_s_bitcmp1_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *scc, uint32_t a, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return bit<1>(flags, instruction_flags, scc, a, b);
}

int goc_rdna4_s_bitcmp0_b64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *scc, uint64_t a, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return bit<0>(flags, instruction_flags, scc, a, b);
}

int goc_rdna4_s_bitcmp1_b64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *scc, uint64_t a, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return bit<1>(flags, instruction_flags, scc, a, b);
}

int goc_rdna4_s_cmp_eq_u64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *scc, uint64_t a, uint64_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return integer<false, 2>(flags, instruction_flags, scc, a, b);
}

int goc_rdna4_s_cmp_lg_u64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *scc, uint64_t a, uint64_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return integer<false, 5>(flags, instruction_flags, scc, a, b);
}

int goc_rdna4_s_cmp_lt_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *scc, uint32_t a, uint32_t b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return floating<32, 1>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

int goc_rdna4_s_cmp_lt_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *scc, uint32_t a, uint32_t b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return floating<16, 1>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

int goc_rdna4_s_cmp_eq_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *scc, uint32_t a, uint32_t b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return floating<32, 2>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

int goc_rdna4_s_cmp_eq_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *scc, uint32_t a, uint32_t b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return floating<16, 2>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

int goc_rdna4_s_cmp_le_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *scc, uint32_t a, uint32_t b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return floating<32, 3>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

int goc_rdna4_s_cmp_le_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *scc, uint32_t a, uint32_t b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return floating<16, 3>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

int goc_rdna4_s_cmp_gt_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *scc, uint32_t a, uint32_t b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return floating<32, 4>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

int goc_rdna4_s_cmp_gt_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *scc, uint32_t a, uint32_t b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return floating<16, 4>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

int goc_rdna4_s_cmp_lg_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *scc, uint32_t a, uint32_t b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return floating<32, 5>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

int goc_rdna4_s_cmp_lg_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *scc, uint32_t a, uint32_t b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return floating<16, 5>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

int goc_rdna4_s_cmp_ge_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *scc, uint32_t a, uint32_t b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return floating<32, 6>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

int goc_rdna4_s_cmp_ge_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *scc, uint32_t a, uint32_t b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return floating<16, 6>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

int goc_rdna4_s_cmp_o_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                          uint32_t *scc, uint32_t a, uint32_t b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return floating<32, 7>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

int goc_rdna4_s_cmp_o_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                          uint32_t *scc, uint32_t a, uint32_t b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return floating<16, 7>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

int goc_rdna4_s_cmp_u_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                          uint32_t *scc, uint32_t a, uint32_t b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return floating<32, 8>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

int goc_rdna4_s_cmp_u_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                          uint32_t *scc, uint32_t a, uint32_t b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return floating<16, 8>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

int goc_rdna4_s_cmp_nge_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *scc, uint32_t a, uint32_t b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return floating<32, 9>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

int goc_rdna4_s_cmp_nge_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *scc, uint32_t a, uint32_t b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return floating<16, 9>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

int goc_rdna4_s_cmp_nlg_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *scc, uint32_t a, uint32_t b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return floating<32, 10>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

int goc_rdna4_s_cmp_nlg_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *scc, uint32_t a, uint32_t b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return floating<16, 10>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

int goc_rdna4_s_cmp_ngt_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *scc, uint32_t a, uint32_t b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return floating<32, 11>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

int goc_rdna4_s_cmp_ngt_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *scc, uint32_t a, uint32_t b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return floating<16, 11>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

int goc_rdna4_s_cmp_nle_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *scc, uint32_t a, uint32_t b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return floating<32, 12>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

int goc_rdna4_s_cmp_nle_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *scc, uint32_t a, uint32_t b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return floating<16, 12>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

int goc_rdna4_s_cmp_neq_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *scc, uint32_t a, uint32_t b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return floating<32, 13>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

int goc_rdna4_s_cmp_neq_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *scc, uint32_t a, uint32_t b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return floating<16, 13>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

int goc_rdna4_s_cmp_nlt_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *scc, uint32_t a, uint32_t b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return floating<32, 14>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

int goc_rdna4_s_cmp_nlt_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *scc, uint32_t a, uint32_t b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  return floating<16, 14>(flags, instruction_flags, scc, a, b, excp_flag_user);
}

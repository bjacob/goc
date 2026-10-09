// SPDX-License-Identifier: MIT

// Models follow rocjitsu generated/shared/execute_shared.h, verified on GFX1201.

#include "goc/goc.h"
#include "internal.h"
#include "rdna4_bits.h"

#include <cstring>
#include <stdint.h>

int goc_rdna4_s_and_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = a & b;
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_and_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                        uint64_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t result = a & b;
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_or_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = a | b;
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_or_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                       uint64_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t result = a | b;
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_xor_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = a ^ b;
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_xor_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                        uint64_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t result = a ^ b;
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_nand_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         uint32_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = ~(a & b);
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_nand_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                         uint64_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t result = ~(a & b);
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_nor_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = ~(a | b);
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_nor_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                        uint64_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t result = ~(a | b);
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_xnor_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         uint32_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = ~(a ^ b);
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_xnor_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                         uint64_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t result = ~(a ^ b);
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_and_not1_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                             uint32_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = a & ~b;
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_and_not1_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                             uint64_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t result = a & ~b;
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_or_not1_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                            uint32_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = a | ~b;
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_or_not1_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                            uint64_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t result = a | ~b;
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_not_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = ~a;
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_not_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                        uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t result = ~a;
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_brev_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = goc::bit_reverse(a);
  *d = result;
  return GOC_SUCCESS;
}

int goc_rdna4_s_brev_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t result = goc::bit_reverse(a);
  *d = result;
  return GOC_SUCCESS;
}

int goc_rdna4_s_lshl_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         uint32_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  unsigned count = b & 31;
  uint32_t result = a << count;
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_lshl_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                         uint32_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  unsigned count = b & 63;
  uint64_t result = a << count;
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_lshr_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         uint32_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  unsigned count = b & 31;
  uint32_t result = a >> count;
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_lshr_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                         uint32_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  unsigned count = b & 63;
  uint64_t result = a >> count;
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_ashr_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         uint32_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  unsigned count = b & 31;
  uint32_t sign = a >> 31;
  uint32_t result = (a >> count) | (count ? (0 - sign) << (32 - count) : 0);
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_ashr_i64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                         uint32_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  unsigned count = b & 63;
  uint64_t sign = a >> 63;
  uint64_t result = (a >> count) | (count ? (0 - sign) << (64 - count) : 0);
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_lshl1_add_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                              uint32_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t wide = (uint64_t(a) << 1) + b;
  uint32_t result = uint32_t(wide), cc = wide > UINT32_MAX;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_lshl2_add_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                              uint32_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t wide = (uint64_t(a) << 2) + b;
  uint32_t result = uint32_t(wide), cc = wide > UINT32_MAX;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_lshl3_add_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                              uint32_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t wide = (uint64_t(a) << 3) + b;
  uint32_t result = uint32_t(wide), cc = wide > UINT32_MAX;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_lshl4_add_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                              uint32_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t wide = (uint64_t(a) << 4) + b;
  uint32_t result = uint32_t(wide), cc = wide > UINT32_MAX;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

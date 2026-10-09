// SPDX-License-Identifier: MIT

// Models follow rocjitsu generated/shared/execute_shared.h, verified on GFX1201.

#include "goc/goc.h"
#include "internal.h"

#include <cstring>
#include <stdint.h>

namespace {

template <typename T> T reverse(T x) {
  const T m1 = T(UINT64_C(0x5555555555555555));
  const T m2 = T(UINT64_C(0x3333333333333333));
  const T m4 = T(UINT64_C(0x0f0f0f0f0f0f0f0f));
  const T m8 = T(UINT64_C(0x00ff00ff00ff00ff));
  x = ((x >> 1) & m1) | ((x & m1) << 1);
  x = ((x >> 2) & m2) | ((x & m2) << 2);
  x = ((x >> 4) & m4) | ((x & m4) << 4);
  x = ((x >> 8) & m8) | ((x & m8) << 8);
  if constexpr (sizeof(T) == 8) {
    const T m16 = UINT64_C(0x0000ffff0000ffff);
    x = ((x >> 16) & m16) | ((x & m16) << 16);
    return (x >> 32) | (x << 32);
  } else {
    return (x >> 16) | (x << 16);
  }
}

} // namespace

int goc_rdna4_s_and_b32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags, uint32_t *d,
                        uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = a & b;
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_and_b64(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags, uint64_t *d,
                        uint64_t a, uint64_t b, uint32_t *scc) {
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t result = a & b;
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_or_b32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags, uint32_t *d,
                       uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = a | b;
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_or_b64(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags, uint64_t *d,
                       uint64_t a, uint64_t b, uint32_t *scc) {
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t result = a | b;
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_xor_b32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags, uint32_t *d,
                        uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = a ^ b;
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_xor_b64(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags, uint64_t *d,
                        uint64_t a, uint64_t b, uint32_t *scc) {
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t result = a ^ b;
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_nand_b32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                         uint32_t *d, uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = ~(a & b);
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_nand_b64(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                         uint64_t *d, uint64_t a, uint64_t b, uint32_t *scc) {
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t result = ~(a & b);
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_nor_b32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags, uint32_t *d,
                        uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = ~(a | b);
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_nor_b64(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags, uint64_t *d,
                        uint64_t a, uint64_t b, uint32_t *scc) {
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t result = ~(a | b);
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_xnor_b32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                         uint32_t *d, uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = ~(a ^ b);
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_xnor_b64(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                         uint64_t *d, uint64_t a, uint64_t b, uint32_t *scc) {
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t result = ~(a ^ b);
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_and_not1_b32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                             uint32_t *d, uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = a & ~b;
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_and_not1_b64(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                             uint64_t *d, uint64_t a, uint64_t b, uint32_t *scc) {
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t result = a & ~b;
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_or_not1_b32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                            uint32_t *d, uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = a | ~b;
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_or_not1_b64(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                            uint64_t *d, uint64_t a, uint64_t b, uint32_t *scc) {
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t result = a | ~b;
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_not_b32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags, uint32_t *d,
                        uint32_t a, uint32_t *scc) {
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = ~a;
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_not_b64(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags, uint64_t *d,
                        uint64_t a, uint32_t *scc) {
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t result = ~a;
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_brev_b32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                         uint32_t *d, uint32_t a) {
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = reverse(a);
  *d = result;
  return GOC_SUCCESS;
}

int goc_rdna4_s_brev_b64(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                         uint64_t *d, uint64_t a) {
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t result = reverse(a);
  *d = result;
  return GOC_SUCCESS;
}

int goc_rdna4_s_lshl_b32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                         uint32_t *d, uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  unsigned count = b & 31;
  uint32_t result = a << count;
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_lshl_b64(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                         uint64_t *d, uint64_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  unsigned count = b & 63;
  uint64_t result = a << count;
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_lshr_b32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                         uint32_t *d, uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  unsigned count = b & 31;
  uint32_t result = a >> count;
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_lshr_b64(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                         uint64_t *d, uint64_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  unsigned count = b & 63;
  uint64_t result = a >> count;
  uint32_t cc = result != 0;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_ashr_i32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                         uint32_t *d, uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
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

int goc_rdna4_s_ashr_i64(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                         uint64_t *d, uint64_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
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

int goc_rdna4_s_lshl1_add_u32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                              uint32_t *d, uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t wide = (uint64_t(a) << 1) + b;
  uint32_t result = uint32_t(wide), cc = wide > UINT32_MAX;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_lshl2_add_u32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                              uint32_t *d, uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t wide = (uint64_t(a) << 2) + b;
  uint32_t result = uint32_t(wide), cc = wide > UINT32_MAX;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_lshl3_add_u32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                              uint32_t *d, uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t wide = (uint64_t(a) << 3) + b;
  uint32_t result = uint32_t(wide), cc = wide > UINT32_MAX;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_lshl4_add_u32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                              uint32_t *d, uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t wide = (uint64_t(a) << 4) + b;
  uint32_t result = uint32_t(wide), cc = wide > UINT32_MAX;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

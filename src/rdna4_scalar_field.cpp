// SPDX-License-Identifier: MIT

// Models borrowed from rocjitsu generated/shared/execute_shared.h and verified
// against GFX1201, including clipped BFE widths and count sentinel values.

#include "goc/goc.h"
#include "internal.h"
#include "rdna4_bits.h"

#include <algorithm>
#include <cstring>
#include <stdint.h>

namespace {

template <typename T, bool Signed> T extract(T a, uint32_t field) {
  const unsigned bits = sizeof(T) * 8;
  unsigned offset = field & (bits - 1), width = std::min((field >> 16) & 127, bits - offset);
  if (!width)
    return 0;
  T mask = width == bits ? ~T(0) : (T(1) << width) - 1;
  T result = (a >> offset) & mask;
  if constexpr (Signed)
    if (result & (T(1) << (width - 1)))
      result |= ~mask;
  return result;
}

} // namespace

int goc_rdna4_s_bfe_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = extract<uint32_t, false>(a, b);
  *d = result;
  uint32_t cc = result != 0;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_bfe_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = extract<uint32_t, true>(a, b);
  *d = result;
  uint32_t cc = result != 0;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_bfe_u64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                        uint32_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t result = extract<uint64_t, false>(a, b);
  *d = result;
  uint32_t cc = result != 0;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_bfe_i64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                        uint32_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t result = extract<uint64_t, true>(a, b);
  *d = result;
  uint32_t cc = result != 0;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_bfm_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  unsigned width = a & 31, offset = b & 31;
  uint32_t result = (((uint32_t(1) << width) - 1) << offset);
  *d = result;
  return GOC_SUCCESS;
}

int goc_rdna4_s_bfm_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint32_t a,
                        uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  unsigned width = a & 63, offset = b & 63;
  uint64_t result = (((uint64_t(1) << width) - 1) << offset);
  *d = result;
  return GOC_SUCCESS;
}

int goc_rdna4_s_bcnt0_i32_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                              uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = goc::bit_population(uint32_t(~a));
  *d = result;
  uint32_t cc = result != 0;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_bcnt0_i32_b64(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint64_t a,
                              uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = goc::bit_population(uint64_t(~a));
  *d = result;
  uint32_t cc = result != 0;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_bcnt1_i32_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                              uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = goc::bit_population(uint32_t(a));
  *d = result;
  uint32_t cc = result != 0;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_bcnt1_i32_b64(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint64_t a,
                              uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = goc::bit_population(uint64_t(a));
  *d = result;
  uint32_t cc = result != 0;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_ctz_i32_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = goc::bit_count_zero<true>(a);
  *d = result;
  return GOC_SUCCESS;
}

int goc_rdna4_s_ctz_i32_b64(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint64_t a) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = goc::bit_count_zero<true>(a);
  *d = result;
  return GOC_SUCCESS;
}

int goc_rdna4_s_clz_i32_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = goc::bit_count_zero<false>(a);
  *d = result;
  return GOC_SUCCESS;
}

int goc_rdna4_s_clz_i32_u64(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint64_t a) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = goc::bit_count_zero<false>(a);
  *d = result;
  return GOC_SUCCESS;
}

int goc_rdna4_s_cls_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  a ^= uint32_t(0) - (a >> 31);
  uint32_t result = goc::bit_count_zero<false>(a);
  *d = result;
  return GOC_SUCCESS;
}

int goc_rdna4_s_cls_i32_i64(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint64_t a) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  a ^= uint64_t(0) - (a >> 63);
  uint32_t result = goc::bit_count_zero<false>(a);
  *d = result;
  return GOC_SUCCESS;
}

int goc_rdna4_s_bitset0_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = *d & ~(uint32_t(1) << (b & 31));
  *d = result;
  return GOC_SUCCESS;
}

int goc_rdna4_s_bitset0_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t result = *d & ~(uint64_t(1) << (b & 63));
  *d = result;
  return GOC_SUCCESS;
}

int goc_rdna4_s_bitset1_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = *d | (uint32_t(1) << (b & 31));
  *d = result;
  return GOC_SUCCESS;
}

int goc_rdna4_s_bitset1_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t result = *d | (uint64_t(1) << (b & 63));
  *d = result;
  return GOC_SUCCESS;
}

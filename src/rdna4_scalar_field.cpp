// SPDX-License-Identifier: MIT

// Models borrowed from rocjitsu generated/shared/execute_shared.h and verified
// against GFX1201, including clipped BFE widths and count sentinel values.

#include "goc/goc.h"
#include "internal.h"

#include <algorithm>
#include <cstring>
#include <stdint.h>

namespace {

template <typename T> uint32_t population(T x) {
  x -= (x >> 1) & T(UINT64_C(0x5555555555555555));
  x = (x & T(UINT64_C(0x3333333333333333))) + ((x >> 2) & T(UINT64_C(0x3333333333333333)));
  x = (x + (x >> 4)) & T(UINT64_C(0x0f0f0f0f0f0f0f0f));
  x += x >> 8;
  x += x >> 16;
  if constexpr (sizeof(T) == 8)
    x += x >> 32;
  return uint32_t(x) & 127;
}

template <bool Trailing, typename T> uint32_t count_zero(T a) {
  if (!a)
    return UINT32_MAX;
#if defined(__GNUC__) || defined(__clang__)
  if constexpr (Trailing) {
    if constexpr (sizeof(T) == 8)
      return __builtin_ctzll(a);
    else
      return __builtin_ctz(a);
  } else {
    if constexpr (sizeof(T) == 8)
      return __builtin_clzll(a);
    else
      return __builtin_clz(a);
  }
#else
  uint32_t n = 0;
  if constexpr (Trailing) {
    while (!(a & 1)) {
      ++n;
      a >>= 1;
    }
  } else {
    const T sign = T(1) << (sizeof(T) * 8 - 1);
    while (!(a & sign)) {
      ++n;
      a <<= 1;
    }
  }
  return n;
#endif
}

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

int goc_rdna4_s_bfe_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                        uint32_t a, uint32_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = extract<uint32_t, false>(a, b);
  *d = result;
  uint32_t cc = result != 0;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_bfe_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                        uint32_t a, uint32_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = extract<uint32_t, true>(a, b);
  *d = result;
  uint32_t cc = result != 0;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_bfe_u64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint64_t *d,
                        uint64_t a, uint32_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t result = extract<uint64_t, false>(a, b);
  *d = result;
  uint32_t cc = result != 0;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_bfe_i64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint64_t *d,
                        uint64_t a, uint32_t b, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t result = extract<uint64_t, true>(a, b);
  *d = result;
  uint32_t cc = result != 0;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_bfm_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                        uint32_t a, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  unsigned width = a & 31, offset = b & 31;
  uint32_t result = (((uint32_t(1) << width) - 1) << offset);
  *d = result;
  return GOC_SUCCESS;
}

int goc_rdna4_s_bfm_b64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint64_t *d,
                        uint32_t a, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  unsigned width = a & 63, offset = b & 63;
  uint64_t result = (((uint64_t(1) << width) - 1) << offset);
  *d = result;
  return GOC_SUCCESS;
}

int goc_rdna4_s_bcnt0_i32_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, uint32_t a, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = population(uint32_t(~a));
  *d = result;
  uint32_t cc = result != 0;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_bcnt0_i32_b64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, uint64_t a, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = population(uint64_t(~a));
  *d = result;
  uint32_t cc = result != 0;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_bcnt1_i32_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, uint32_t a, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = population(uint32_t(a));
  *d = result;
  uint32_t cc = result != 0;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_bcnt1_i32_b64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, uint64_t a, uint32_t *scc) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = population(uint64_t(a));
  *d = result;
  uint32_t cc = result != 0;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

int goc_rdna4_s_ctz_i32_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, uint32_t a) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = count_zero<true>(a);
  *d = result;
  return GOC_SUCCESS;
}

int goc_rdna4_s_ctz_i32_b64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, uint64_t a) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = count_zero<true>(a);
  *d = result;
  return GOC_SUCCESS;
}

int goc_rdna4_s_clz_i32_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, uint32_t a) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = count_zero<false>(a);
  *d = result;
  return GOC_SUCCESS;
}

int goc_rdna4_s_clz_i32_u64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, uint64_t a) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = count_zero<false>(a);
  *d = result;
  return GOC_SUCCESS;
}

int goc_rdna4_s_cls_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                        uint32_t a) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  a ^= uint32_t(0) - (a >> 31);
  uint32_t result = count_zero<false>(a);
  *d = result;
  return GOC_SUCCESS;
}

int goc_rdna4_s_cls_i32_i64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, uint64_t a) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  a ^= uint64_t(0) - (a >> 63);
  uint32_t result = count_zero<false>(a);
  *d = result;
  return GOC_SUCCESS;
}

int goc_rdna4_s_bitset0_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = *d & ~(uint32_t(1) << (b & 31));
  *d = result;
  return GOC_SUCCESS;
}

int goc_rdna4_s_bitset0_b64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint64_t *d, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t result = *d & ~(uint64_t(1) << (b & 63));
  *d = result;
  return GOC_SUCCESS;
}

int goc_rdna4_s_bitset1_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint32_t result = *d | (uint32_t(1) << (b & 31));
  *d = result;
  return GOC_SUCCESS;
}

int goc_rdna4_s_bitset1_b64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint64_t *d, uint32_t b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  (void)exec_mask;
  if (int error = goc::validate(flags, instruction_flags, true))
    return error;
  uint64_t result = *d | (uint64_t(1) << (b & 63));
  *d = result;
  return GOC_SUCCESS;
}

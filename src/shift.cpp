// SPDX-License-Identifier: MIT

#include "shift.h"
#include "dpp.h"
#include "goc/goc.h"
#include "internal.h"

#include <stdint.h>

namespace {

template <int Bits, goc::Shift Op>
int shift(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
          const uint32_t *const *a, const uint32_t *const *b) {
  if (mode >> 32) {
    if constexpr (Bits == 32) {
      return goc::execute_dpp(
          flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
            return shift<Bits, Op>(flags, exec_mask, uint32_t(mode), d, source, b);
          });
    } else {
      return GOC_ERROR_INVALID_FLAGS;
    }
  }

  if (int error = goc::validate(flags, mode, true))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::shift_x86_64_v3<Bits, Op>(exec_mask, d, a[0], b);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[Bits / 32][32];
  for (int lane = 0; lane < 32; ++lane) {
    unsigned count = a[0][lane] & (Bits - 1);
    uint64_t value = b[0][lane];
    if constexpr (Bits == 64)
      value |= uint64_t(b[1][lane]) << 32;
    if constexpr (Op == goc::Shift::Left)
      value <<= count;
    if constexpr (Op == goc::Shift::LogicalRight)
      value >>= count;
    if constexpr (Op == goc::Shift::ArithmeticRight) {
      const uint64_t width_mask = Bits == 32 ? UINT32_MAX : UINT64_MAX;
      const uint64_t sign = value >> (Bits - 1) ? width_mask : 0;
      // Complement negative values around a logical shift to supply sign bits
      // without implementation-defined signed right shifts.
      value = ((value ^ sign) >> count) ^ sign;
    }
    result[0][lane] = uint32_t(value);
    if constexpr (Bits == 64)
      result[1][lane] = uint32_t(value >> 32);
  }
  for (int reg = 0; reg < Bits / 32; ++reg)
    for (int lane = 0; lane < 32; ++lane)
      if ((exec_mask >> lane) & 1)
        d[reg][lane] = result[reg][lane];
  return GOC_SUCCESS;
}

template <int Bits, goc::Shift Op>
int shift_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                 const uint32_t *const *a, const uint32_t *const *b) {
  if (int error = shift<Bits, Op>(flags, 0, mode, nullptr, nullptr, nullptr))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
  for (unsigned half = 0; half < 2; ++half) {
    uint64_t half_mode = mode;
    if (mode & GOC_DPP16) {
      uint64_t rows = (mode & GOC_DPP_ROW_MASK) >> GOC_DPP_ROW_SHIFT;
      half_mode = (mode & ~GOC_DPP_ROW_MASK) | (((rows >> (half * 2)) & 3) << GOC_DPP_ROW_SHIFT);
    }
    const uint32_t *counts = a[0] + half * 32, *values[Bits / 32];
    uint32_t *dest[Bits / 32];
    for (unsigned word = 0; word < Bits / 32; ++word) {
      values[word] = b[word] + half * 32;
      dest[word] = d[word] + half * 32;
    }
    if (int error = shift<Bits, Op>(flags, uint32_t(exec_mask >> (half * 32)), half_mode, dest,
                                    &counts, values))
      return error;
  }
  return GOC_SUCCESS;
}

} // namespace

int goc_v_lshlrev_b32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                      const uint32_t *const *a, const uint32_t *const *b) {
  return shift<32, goc::Shift::Left>(flags, exec_mask, mode, d, a, b);
}

int goc_v_lshrrev_b32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                      const uint32_t *const *a, const uint32_t *const *b) {
  return shift<32, goc::Shift::LogicalRight>(flags, exec_mask, mode, d, a, b);
}

int goc_v_ashrrev_i32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                      const uint32_t *const *a, const uint32_t *const *b) {
  return shift<32, goc::Shift::ArithmeticRight>(flags, exec_mask, mode, d, a, b);
}

int goc_v_lshlrev_b64(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                      const uint32_t *const *a, const uint32_t *const *b) {
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return shift<64, goc::Shift::Left>(flags, exec_mask, mode, d, a, b);
}

int goc_v_lshrrev_b64(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                      const uint32_t *const *a, const uint32_t *const *b) {
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return shift<64, goc::Shift::LogicalRight>(flags, exec_mask, mode, d, a, b);
}

int goc_v_ashrrev_i64(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                      const uint32_t *const *a, const uint32_t *const *b) {
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return shift<64, goc::Shift::ArithmeticRight>(flags, exec_mask, mode, d, a, b);
}

int goc_v_lshlrev_b32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b) {
  return shift_wave64<32, goc::Shift::Left>(flags, exec_mask, mode, d, a, b);
}

int goc_v_lshrrev_b32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b) {
  return shift_wave64<32, goc::Shift::LogicalRight>(flags, exec_mask, mode, d, a, b);
}

int goc_v_ashrrev_i32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b) {
  return shift_wave64<32, goc::Shift::ArithmeticRight>(flags, exec_mask, mode, d, a, b);
}

int goc_v_lshlrev_b64_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b) {
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return shift_wave64<64, goc::Shift::Left>(flags, exec_mask, mode, d, a, b);
}

int goc_v_lshrrev_b64_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b) {
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return shift_wave64<64, goc::Shift::LogicalRight>(flags, exec_mask, mode, d, a, b);
}

int goc_v_ashrrev_i64_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b) {
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return shift_wave64<64, goc::Shift::ArithmeticRight>(flags, exec_mask, mode, d, a, b);
}

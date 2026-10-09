// SPDX-License-Identifier: MIT

// Table extraction follows rocjitsu's execute_v_trig_preop_f64_vop3.
// GFX1201 captures establish a 1184-bit table and pre-scaling OMOD flush.

#include "rdna4_trig_preop.h"
#include "goc/goc.h"
#include "internal.h"

#include <algorithm>
#include <array>
#include <stdint.h>

namespace {

constexpr std::array<uint64_t, 2370> make_table() {
  constexpr uint32_t chunks[] = {
      0xA2F983, 0x6E4E44, 0x1529FC, 0x2757D1, 0xF534DD, 0xC0DB62, 0x95993C, 0x439041, 0xFE5163,
      0xABDEBB, 0xC561B7, 0x246E3A, 0x424DD2, 0xE00649, 0x2EEA09, 0xD1921C, 0xFE1DEB, 0x1CB129,
      0xA73EE8, 0x8235F5, 0x2EBB44, 0x84E99C, 0x7026B4, 0x5F7E41, 0x3991D6, 0x398353, 0x39F49C,
      0x845F8B, 0xBDF928, 0x3B1FF8, 0x97FFDE, 0x05980F, 0xEF2F11, 0x8B5A0A, 0x6D1F6D, 0x367ECF,
      0x27CB09, 0xB74F46, 0x3F669E, 0x5FEA2D, 0x7527BA, 0xC7EBE5, 0xF17B3D, 0x0739F7, 0x8A5292,
      0xEA6BFB, 0x5FB11F, 0x8D5D08, 0x560330, 0x46FC7B};
  std::array<uint64_t, 2370> table{};
  for (unsigned bank = 0; bank < 2; ++bank) {
    for (unsigned shift = 0; shift < 1184; ++shift) {
      uint64_t significand = 0;
      for (unsigned bit = shift; bit < shift + 53; ++bit)
        significand =
            (significand << 1) | (bit < 1184 ? (chunks[bit / 24] >> (23 - bit % 24)) & 1 : 0);
      if (!significand)
        continue;
      int scale = -53 - int(shift) + 128 * int(bank);
      unsigned leading = 0;
      for (uint64_t v = significand; v > 1; v >>= 1)
        ++leading;
      int exponent = int(leading) + scale;
      uint64_t value = 0;
      if (exponent < -1022) {
        int adjustment = scale + 1074;
        value = adjustment >= 0    ? significand << adjustment
                : adjustment > -64 ? significand >> -adjustment
                                   : 0;
      } else {
        value = (uint64_t(exponent + 1023) << 52) |
                ((significand << (52 - leading)) & 0xfffffffffffffULL);
      }
      table[bank * 1185 + shift] = value;
    }
  }
  return table;
}

} // namespace

namespace goc {

constexpr std::array<uint64_t, 2370> trig_preop_table = make_table();

} // namespace goc

int goc_rdna4_v_trig_preop_f64(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_EXCEPTIONS;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  const uint32_t known =
      GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_2 | GOC_ALU_OMOD_4 | GOC_ALU_CLAMP;
  if (int error = goc::validate(flags, mode & ~known, true))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::trig_preop_x86_64_v4(exec_mask, mode, d, a[1], b[0]);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::trig_preop_x86_64_v3(exec_mask, mode, d, a[1], b[0]);
    return GOC_SUCCESS;
  }
#endif
  uint64_t result[32];
  for (unsigned lane = 0; lane < 32; ++lane) {
    unsigned exponent = (a[1][lane] >> 20) & 2047;
    unsigned shift = 53 * (b[0][lane] & 31) + (exponent > 1077 ? exponent - 1077 : 0);
    uint64_t value = goc::trig_preop_table[(exponent >= 1968 ? 1185 : 0) + std::min(shift, 1184u)];
    unsigned omod = (mode >> 6) & 3;
    if (omod) {
      if (!(value >> 52))
        value = 0;
      else {
        value += uint64_t(int64_t(omod == 3 ? -1 : int(omod)) * (INT64_C(1) << 52));
        if (!(value >> 52))
          value = 0;
      }
    }
    if (mode & GOC_ALU_CLAMP)
      value = std::min<uint64_t>(value, 0x3ff0000000000000ULL);
    result[lane] = value;
  }
  for (unsigned reg = 0; reg < 2; ++reg)
    for (unsigned lane = 0; lane < 32; ++lane)
      if ((exec_mask >> lane) & 1)
        d[reg][lane] = uint32_t(result[lane] >> (32 * reg));
  return GOC_SUCCESS;
}

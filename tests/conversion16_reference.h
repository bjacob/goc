// SPDX-License-Identifier: MIT

#pragma once

#include "conversion64_reference.h"
#include "goc/goc.h"

#include <stdint.h>

namespace goc_test {

inline const char *const conversion16_names[] = {"v_cvt_f16_i16", "v_cvt_f16_u16", "v_cvt_i16_f16",
                                                 "v_cvt_u16_f16", "v_cvt_f16_f32", "v_cvt_f32_f16"};

inline unsigned conversion16_modes(int op) { return op < 2 ? 32 : op < 4 ? 128 : 64; }

inline uint32_t conversion16_mode(int op, unsigned variant) {
  const uint32_t bits[] = {GOC_ALU_OMOD_2,
                           GOC_ALU_OMOD_4,
                           GOC_ALU_CLAMP,
                           op < 2 ? 0 : GOC_ALU_ABS_A,
                           op < 2 ? 0 : GOC_ALU_NEG_A,
                           op == 4 ? 0 : GOC_ALU_HIGH_A,
                           op == 5 ? 0 : GOC_ALU_HIGH_D};
  uint32_t mode = 0;
  for (uint32_t bit : bits)
    if (bit) {
      if (variant & 1)
        mode |= bit;
      variant >>= 1;
    }
  return mode;
}

inline uint16_t conversion16_narrow(uint32_t value, bool saturate) {
  uint16_t result = uint16_t(conversion_reencode(value, 23, 127, 10, 15));
  if (saturate && (value & 0x7fffffff) < 0x7f800000 && (result & 0x7fff) == 0x7c00)
    --result;
  return result;
}

inline uint32_t conversion16_reference(int op, uint32_t raw, uint32_t original, uint32_t mode,
                                       bool saturate) {
  uint16_t half = uint16_t(raw >> (mode & GOC_ALU_HIGH_A ? 16 : 0));
  uint32_t value;
  if (op < 2) {
    bool negative = op == 0 && (half >> 15);
    uint16_t magnitude = negative ? uint16_t(0u - half) : half;
    value = uint32_t(conversion_encode(negative, magnitude, 0, 23, 127));
  } else {
    value = op == 4 ? raw : uint32_t(conversion_reencode(half, 10, 15, 23, 127));
    if (mode & GOC_ALU_ABS_A)
      value &= 0x7fffffff;
    if (mode & GOC_ALU_NEG_A)
      value ^= 0x80000000;
  }
  uint32_t result;
  if (op == 2 || op == 3) {
    // Decode the exact binary integer part; no host floating-point casts.
    unsigned exponent = (value >> 23) & 255;
    bool negative = value >> 31;
    uint32_t fraction = value & 0x7fffff;
    uint32_t limit = op == 3 ? 65535 : negative ? 32768 : 32767;
    if ((exponent == 255 && fraction) || (op == 3 && negative) || exponent < 127)
      result = 0;
    else {
      uint32_t magnitude = exponent >= 143 ? limit : (0x800000 | fraction) >> (150 - exponent);
      if (magnitude > limit)
        magnitude = limit;
      result = negative ? uint16_t(0u - magnitude) : magnitude;
    }
  } else {
    const int scales[] = {0, 1, 2, -1};
    unsigned omod = (mode >> 6) & 3;
    if (op == 5) {
      result = uint32_t(conversion_reencode(value, 23, 127, 23, 127, scales[omod]));
      if (omod && (result & 0x7fffffff) == 0)
        result = 0;
      if (mode & GOC_ALU_CLAMP)
        result = (result & 0x80000000) || result > 0x7f800000 ? 0
                 : result > 0x3f800000                        ? 0x3f800000
                                                              : result;
    } else {
      result = conversion16_narrow(value, saturate);
      if (omod) {
        unsigned magnitude = result & 0x7fff;
        if (magnitude < 0x400 || (magnitude == 0x400 && (value & 0x7fffffff) < 0x387ff000))
          result = 0;
        else if (omod == 3 && magnitude < 0x800)
          result &= 0x8000;
        else {
          uint32_t scaled = uint32_t(conversion_reencode(result, 10, 15, 23, 127, scales[omod]));
          result = conversion16_narrow(scaled, saturate);
        }
      }
      if (mode & GOC_ALU_CLAMP)
        result = (result & 0x8000) || result > 0x7c00 ? 0 : result > 0x3c00 ? 0x3c00 : result;
    }
  }
  if (op == 5)
    return result;
  unsigned shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  return (original & ~(uint32_t(65535) << shift)) | (result << shift);
}

inline bool conversion16_equal(int op, uint32_t actual, uint32_t expected, uint32_t mode) {
  if (actual == expected)
    return true;
  if (op == 2 || op == 3)
    return false;
  if (op != 5) {
    unsigned shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
    uint32_t keep = ~(uint32_t(65535) << shift);
    if ((actual & keep) != (expected & keep))
      return false;
    actual = uint16_t(actual >> shift);
    expected = uint16_t(expected >> shift);
    return (actual & 0x7fff) > 0x7c00 && (expected & 0x7fff) > 0x7c00 && (actual & 0x200) &&
           (expected & 0x200);
  }
  return (actual & 0x7fffffff) > 0x7f800000 && (expected & 0x7fffffff) > 0x7f800000 &&
         (actual & 0x400000) && (expected & 0x400000);
}

} // namespace goc_test

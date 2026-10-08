// SPDX-License-Identifier: MIT

#include "float_formats.h"
#include "goc_common.h"
#include "goc_rdna4.h"
#include "internal.h"
#include "packed16.h"
#include "rdna4_dot.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <stdint.h>

namespace {

// Physical packing follows rocjitsu shared/mma_exec.h: each lane supplies
// eight (wave32) or four (wave64) consecutive K elements. Output lanes select
// columns and row groups.
template <bool Bf16, int WaveSize = 32, bool Packed = false>
int wmma(uint64_t flags, uint64_t mask, uint32_t instruction_flags, uint32_t *const *d,
         const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (int error = goc::validate(flags, instruction_flags & ~UINT32_C(63), true))
    return error;

#if defined(GOC_HAVE_AVX512BF16)
  if constexpr (Bf16 && WaveSize == 32 && !Packed) {
    if ((flags & GOC_CPU_MASK) >= GOC_CPU_ZEN4 &&
        (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_EXACT_EMPIRICAL && instruction_flags == 0) {
      // DPBF16 flushes denormals independently of MXCSR. Retain the scalar
      // path for subnormal factors or accumulators and special values.
      bool ordinary = true;
      int min_exp[2] = {255, 255}, max_exp[2] = {0, 0};
      for (int reg = 0; reg < 4; ++reg)
        for (int lane = 0; lane < 32; ++lane)
          for (int shift : {0, 16}) {
            for (int operand = 0; operand < 2; ++operand) {
              uint32_t word = operand ? b[reg][lane] : a[reg][lane];
              uint16_t bits = uint16_t(word >> shift);
              int exp = (bits >> 7) & 255;
              if (bits & 0x7fff) {
                min_exp[operand] = std::min(min_exp[operand], exp);
                max_exp[operand] = std::max(max_exp[operand], exp);
              }
              if (exp == 255 || (exp == 0 && (bits & 127)))
                ordinary = false;
            }
          }
      for (int reg = 0; reg < 8; ++reg)
        for (int lane = 0; lane < 32; ++lane) {
          uint32_t bits = c[reg][lane];
          int exp = (bits >> 23) & 255;
          if (exp == 255 || (exp == 0 && (bits & 0x7fffff)))
            ordinary = false;
        }

      // Keep products safely inside FP32's normal range as well: normal
      // factors alone do not rule out product underflow or overflow.
      ordinary &= min_exp[0] + min_exp[1] >= 128 && max_exp[0] + max_exp[1] <= 380;
      if (ordinary) {
        goc::wmma_avx512bf16(static_cast<uint32_t>(mask), d, a, b, c);
        return GOC_SUCCESS;
      }
    }
  }
#endif

  constexpr int OutputRegs = 256 / WaveSize;
  constexpr int KPerLane = 256 / WaveSize;
  constexpr int DestinationRegs = Packed ? OutputRegs / 2 : OutputRegs;
  uint32_t result[DestinationRegs][WaveSize] = {};
  const auto read_bits = [instruction_flags](const uint32_t *const *v, int index, int k,
                                             int operand) {
    uint16_t value = uint16_t(v[(k % KPerLane) / 2][index + 16 * (k / KPerLane)] >> (16 * (k % 2)));
    if ((instruction_flags >> (operand + 3 * (k % 2))) & 1)
      value ^= 0x8000;
    return value;
  };
  for (int row = 0; row < 16; ++row)
    for (int col = 0; col < 16; ++col) {
      int lane = col + 16 * (row / 8) + (WaveSize == 64 ? 32 * ((row / 4) % 2) : 0);
      int reg = row % OutputRegs;
      if (!((mask >> lane) & 1))
        continue;
      uint32_t c_bits = c[Packed ? reg / 2 : reg][lane];
      if constexpr (Packed)
        c_bits = (c_bits >> (16 * (reg % 2))) & 0xffff;
      if (instruction_flags & GOC_WMMA_ABS_C)
        c_bits &= Packed ? 0x7fff : 0x7fffffff;
      if (instruction_flags & GOC_WMMA_NEG_C)
        c_bits ^= Packed ? 0x8000 : 0x80000000;
      if (Packed || (flags & GOC_SEMANTICS_MASK) == GOC_SEMANTICS_EXACT_EMPIRICAL) {
        uint32_t acc = c_bits;
        for (int k = 0; k < 16; k += 4) {
          std::array<uint16_t, 4> left, right;
          for (int j = 0; j < 4; ++j) {
            // GFX12 wave32 processes physical K chunks in order 0,8,4,12;
            // wave64 processes them in order 0,4,8,12.
            int logical = k + j;
            int physical = WaveSize == 32
                               ? (logical & 3) | ((logical & 4) << 1) | ((logical & 8) >> 1)
                               : logical;
            left[j] = read_bits(a, row, physical, 0);
            right[j] = read_bits(b, col, physical, 1);
          }
          if constexpr (Packed)
            acc = goc::packed16::widen<Bf16>(uint16_t(acc));
          acc = goc::gfx12_dot_bits<Bf16, 4, Packed>(left, right, acc, flags & GOC_FP16_OVFL);
        }
        if constexpr (Packed)
          result[reg / 2][lane] |= acc << (16 * (reg % 2));
        else
          result[reg][lane] = acc;
      } else {
        float acc = goc::as_float(c_bits);
        for (int k = 0; k < 16; ++k) {
          auto left = read_bits(a, row, k, 0), right = read_bits(b, col, k, 1);
          float x = Bf16 ? goc::bf16_to_float(left) : goc::f16_to_float(left);
          float y = Bf16 ? goc::bf16_to_float(right) : goc::f16_to_float(right);
          acc = std::fma(x, y, acc);
        }
        result[reg][lane] = goc::as_bits(acc);
      }
    }

  // Delayed stores are necessary even with exact whole-VGPR aliasing: a
  // destination may overwrite sources consumed by a different output lane.
  for (int reg = 0; reg < DestinationRegs; ++reg)
    for (int lane = 0; lane < WaveSize; ++lane)
      if ((mask >> lane) & 1)
        d[reg][lane] = result[reg][lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_wmma_f32_16x16x16_f16(uint64_t flags, uint64_t mask, uint32_t instruction_flags,
                                      uint32_t *const *d, const uint32_t *const *a,
                                      const uint32_t *const *b, const uint32_t *const *c) {
  return wmma<false>(flags, mask, instruction_flags, d, a, b, c);
}

int goc_rdna4_v_wmma_f32_16x16x16_bf16(uint64_t flags, uint64_t mask, uint32_t instruction_flags,
                                       uint32_t *const *d, const uint32_t *const *a,
                                       const uint32_t *const *b, const uint32_t *const *c) {
  return wmma<true>(flags, mask, instruction_flags, d, a, b, c);
}

int goc_rdna4w64_v_wmma_f32_16x16x16_f16(uint64_t flags, uint64_t mask, uint32_t instruction_flags,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b, const uint32_t *const *c) {
  return wmma<false, 64>(flags, mask, instruction_flags, d, a, b, c);
}

int goc_rdna4w64_v_wmma_f32_16x16x16_bf16(uint64_t flags, uint64_t mask, uint32_t instruction_flags,
                                          uint32_t *const *d, const uint32_t *const *a,
                                          const uint32_t *const *b, const uint32_t *const *c) {
  return wmma<true, 64>(flags, mask, instruction_flags, d, a, b, c);
}

int goc_rdna4_v_wmma_f16_16x16x16_f16(uint64_t flags, uint64_t mask, uint32_t instruction_flags,
                                      uint32_t *const *d, const uint32_t *const *a,
                                      const uint32_t *const *b, const uint32_t *const *c) {
  return wmma<false, 32, true>(flags, mask, instruction_flags, d, a, b, c);
}

int goc_rdna4_v_wmma_bf16_16x16x16_bf16(uint64_t flags, uint64_t mask, uint32_t instruction_flags,
                                        uint32_t *const *d, const uint32_t *const *a,
                                        const uint32_t *const *b, const uint32_t *const *c) {
  return wmma<true, 32, true>(flags, mask, instruction_flags, d, a, b, c);
}

int goc_rdna4w64_v_wmma_f16_16x16x16_f16(uint64_t flags, uint64_t mask, uint32_t instruction_flags,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b, const uint32_t *const *c) {
  return wmma<false, 64, true>(flags, mask, instruction_flags, d, a, b, c);
}

int goc_rdna4w64_v_wmma_bf16_16x16x16_bf16(uint64_t flags, uint64_t mask,
                                           uint32_t instruction_flags, uint32_t *const *d,
                                           const uint32_t *const *a, const uint32_t *const *b,
                                           const uint32_t *const *c) {
  return wmma<true, 64, true>(flags, mask, instruction_flags, d, a, b, c);
}

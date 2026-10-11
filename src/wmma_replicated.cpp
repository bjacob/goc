// SPDX-License-Identifier: MIT

#include "float_formats.h"
#include "gfx11_dot2.h"
#include "goc/goc.h"
#include "internal.h"
#include "packed16.h"
#include "simd.h"

#include <algorithm>
#include <cmath>
#include <stdint.h>

namespace {

// RDNA3 layouts follow rocjitsu shared/mma_exec.h's gfx11_wmma_input_loc
// and gfx11_wmma_output_loc_32. Each 16-lane group holds the full A/B tile;
// output rows are interleaved across groups, unlike RDNA4's blocked layout.
template <bool Bf16, int Lanes, bool Packed = false>
int wmma(uint64_t flags, uint64_t mode, uint32_t *const *d, const uint32_t *const *a,
         const uint32_t *const *b, const uint32_t *const *c) {
  const uint64_t known = 63 | (Packed ? GOC_WMMA_HIGH_C_D : 0);
  if (mode & ~known)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, 0, true, goc::matrix_ignored_fp_flags))
    return error;
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3 &&
      (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_EXACT_EMPIRICAL) {
    if constexpr (Packed)
      goc::wmma_packed_x86_64_v3<Bf16, Lanes, true>(uint32_t(mode), flags & GOC_FP16_OVFL, d, a, b,
                                                    c);
    else
      goc::wmma_f32_x86_64_v3<Bf16, Lanes, true>(uint32_t(mode), d, a, b, c);
    return GOC_SUCCESS;
  }
#endif
  constexpr int Groups = Lanes / 16, Registers = 256 / Lanes;
  const unsigned shift = (mode & GOC_WMMA_HIGH_C_D) ? 16 : 0;
  uint32_t result[Registers][Lanes];
  for (int row = 0; row < 16; ++row)
    for (int col = 0; col < 16; ++col) {
      int group = row % Groups, lane = group * 16 + col, reg = row / Groups;
      uint32_t acc = c[reg][lane];
      if constexpr (Packed)
        acc = (acc >> shift) & 65535;
      if (mode & GOC_WMMA_ABS_C)
        acc &= Packed ? 0x7fff : 0x7fffffff;
      if (mode & GOC_WMMA_NEG_C)
        acc ^= Packed ? 0x8000 : 0x80000000;
      float loose = goc::as_float(Packed ? goc::packed16::widen<Bf16>(uint16_t(acc)) : acc);
      for (int k = 0; k < 16; k += 2) {
        uint32_t av = a[k / 2][group * 16 + row];
        uint32_t bv = b[k / 2][group * 16 + col];
        av ^= ((mode & GOC_WMMA_NEG_LO_A) ? 0x8000U : 0) |
              ((mode & GOC_WMMA_NEG_HI_A) ? 0x80000000U : 0);
        bv ^= ((mode & GOC_WMMA_NEG_LO_B) ? 0x8000U : 0) |
              ((mode & GOC_WMMA_NEG_HI_B) ? 0x80000000U : 0);
        if ((flags & GOC_SEMANTICS_MASK) == GOC_SEMANTICS_EXACT_EMPIRICAL) {
          if constexpr (Packed)
            acc = goc::gfx11_dot2_packed16<Bf16>(av, bv, av >> 16, bv >> 16, acc,
                                                 flags & GOC_FP16_OVFL);
          else
            acc = goc::gfx11_dot2_f32<Bf16>(av, bv, av >> 16, bv >> 16, acc);
        } else {
          for (int half = 0; half < 2; ++half) {
            uint16_t x = uint16_t(av >> (16 * half)), y = uint16_t(bv >> (16 * half));
            loose = std::fma(Bf16 ? goc::bf16_to_float(x) : goc::f16_to_float(x),
                             Bf16 ? goc::bf16_to_float(y) : goc::f16_to_float(y), loose);
          }
          if constexpr (Packed) {
            // RDNA3 narrows after every two products, RDNA4 after four.
            acc =
                Bf16 ? goc::as_bits(loose) >> 16 : goc::float_to_f16(loose, flags & GOC_FP16_OVFL);
            loose = goc::as_float(goc::packed16::widen<Bf16>(uint16_t(acc)));
          }
        }
      }
      if constexpr (Packed)
        result[reg][lane] = (d[reg][lane] & ~(65535U << shift)) | (acc << shift);
      else
        result[reg][lane] = (flags & GOC_SEMANTICS_MASK) == GOC_SEMANTICS_EXACT_EMPIRICAL
                                ? acc
                                : goc::as_bits(loose);
    }
  for (int reg = 0; reg < Registers; ++reg)
    for (int lane = 0; lane < Lanes; ++lane)
      d[reg][lane] = result[reg][lane];
  return GOC_SUCCESS;
}

// rocjitsu de4654cfc34: signed C, with saturation after each pair of IU8
// products or group of four IU4 products. Unclamped accumulation wraps.
template <int Bits, int Lanes>
int integer_wmma(uint64_t flags, uint64_t mode, uint32_t *const *d, const uint32_t *const *a,
                 const uint32_t *const *b, const uint32_t *const *c) {
  if (mode & ~(uint64_t(GOC_WMMA_SIGNED_A | GOC_WMMA_SIGNED_B | GOC_WMMA_CLAMP)))
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, 0, true))
    return error;
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::integer_wmma_replicated_x86_64_v3<Bits, Lanes>(uint32_t(mode), d, a, b, c);
    return GOC_SUCCESS;
  }
#endif
  constexpr int Groups = Lanes / 16, Registers = 256 / Lanes, PerReg = 32 / Bits;
  uint32_t result[Registers][Lanes];
  for (int row = 0; row < 16; ++row)
    for (int col = 0; col < 16; ++col) {
      int group = row % Groups, lane = group * 16 + col, reg = row / Groups;
      uint32_t cv = c[reg][lane];
      int64_t acc = int64_t(cv) - ((cv >> 31) ? (1LL << 32) : 0);
      for (int k = 0; k < 16; ++k) {
        int x = (a[k / PerReg][16 * group + row] >> (Bits * (k % PerReg))) & ((1U << Bits) - 1);
        int y = (b[k / PerReg][16 * group + col] >> (Bits * (k % PerReg))) & ((1U << Bits) - 1);
        if ((mode & GOC_WMMA_SIGNED_A) && (x & (1 << (Bits - 1))))
          x -= 1 << Bits;
        if ((mode & GOC_WMMA_SIGNED_B) && (y & (1 << (Bits - 1))))
          y -= 1 << Bits;
        acc += x * y;
        constexpr int clamp_group = Bits == 8 ? 2 : 4;
        if ((mode & GOC_WMMA_CLAMP) && (k + 1) % clamp_group == 0)
          acc = std::clamp(acc, int64_t(-2147483648LL), int64_t(2147483647));
      }
      result[reg][lane] = uint32_t(acc);
    }
  for (int reg = 0; reg < Registers; ++reg)
    for (int lane = 0; lane < Lanes; ++lane)
      d[reg][lane] = result[reg][lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_v_wmma_f32_16x16x16_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                const uint32_t *const *c) {
  return wmma<false, 32>(flags, instruction_flags, d, a, b, c);
}

int goc_v_wmma_f32_16x16x16_bf16(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                                 const uint32_t *const *a, const uint32_t *const *b,
                                 const uint32_t *const *c) {
  return wmma<true, 32>(flags, instruction_flags, d, a, b, c);
}

int goc_v_wmma_f32_16x16x16_f16_wave64(uint64_t flags, uint64_t instruction_flags,
                                       uint32_t *const *d, const uint32_t *const *a,
                                       const uint32_t *const *b, const uint32_t *const *c) {
  return wmma<false, 64>(flags, instruction_flags, d, a, b, c);
}

int goc_v_wmma_f32_16x16x16_bf16_wave64(uint64_t flags, uint64_t instruction_flags,
                                        uint32_t *const *d, const uint32_t *const *a,
                                        const uint32_t *const *b, const uint32_t *const *c) {
  return wmma<true, 64>(flags, instruction_flags, d, a, b, c);
}

int goc_v_wmma_f16_16x16x16_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                const uint32_t *const *c) {
  return wmma<false, 32, true>(flags, instruction_flags, d, a, b, c);
}

int goc_v_wmma_f16_16x16x16_f16_wave64(uint64_t flags, uint64_t instruction_flags,
                                       uint32_t *const *d, const uint32_t *const *a,
                                       const uint32_t *const *b, const uint32_t *const *c) {
  return wmma<false, 64, true>(flags, instruction_flags, d, a, b, c);
}

int goc_v_wmma_bf16_16x16x16_bf16(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                                  const uint32_t *const *a, const uint32_t *const *b,
                                  const uint32_t *const *c) {
  return wmma<true, 32, true>(flags, instruction_flags, d, a, b, c);
}

int goc_v_wmma_bf16_16x16x16_bf16_wave64(uint64_t flags, uint64_t instruction_flags,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b, const uint32_t *const *c) {
  return wmma<true, 64, true>(flags, instruction_flags, d, a, b, c);
}

int goc_v_wmma_i32_16x16x16_iu8(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                const uint32_t *const *c) {
  return integer_wmma<8, 32>(flags, instruction_flags, d, a, b, c);
}

int goc_v_wmma_i32_16x16x16_iu8_wave64(uint64_t flags, uint64_t instruction_flags,
                                       uint32_t *const *d, const uint32_t *const *a,
                                       const uint32_t *const *b, const uint32_t *const *c) {
  return integer_wmma<8, 64>(flags, instruction_flags, d, a, b, c);
}

int goc_v_wmma_i32_16x16x16_iu4(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                const uint32_t *const *c) {
  return integer_wmma<4, 32>(flags, instruction_flags, d, a, b, c);
}

int goc_v_wmma_i32_16x16x16_iu4_wave64(uint64_t flags, uint64_t instruction_flags,
                                       uint32_t *const *d, const uint32_t *const *a,
                                       const uint32_t *const *b, const uint32_t *const *c) {
  return integer_wmma<4, 64>(flags, instruction_flags, d, a, b, c);
}

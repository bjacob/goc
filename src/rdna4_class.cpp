// SPDX-License-Identifier: MIT

// Class bits follow rocjitsu's execute_v_cmp_class_f{16,32,64} bitwise models.

#include "rdna4_class.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_dpp.h"

#include <stdint.h>

namespace {

template <unsigned Bits>
int run(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *d, const uint32_t *const *a,
        const uint32_t *const *b) {
  if (mode >> 32) {
    if constexpr (Bits == 64)
      return GOC_ERROR_INVALID_FLAGS;
    if (!(mode & (GOC_DPP8 | GOC_DPP16)) || !goc::valid_dpp(mode))
      return GOC_ERROR_INVALID_FLAGS;
  }
  const uint32_t known =
      GOC_ALU_ABS_A | GOC_ALU_NEG_A | (Bits == 16 ? GOC_ALU_HIGH_A | GOC_ALU_HIGH_B : 0);
  if (int error = goc::validate(flags, uint32_t(mode) & ~known, true,
                                GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS))
    return error;
  if (!exec_mask) {
    *d = 0;
    return GOC_SUCCESS;
  }
  if constexpr (Bits != 64) {
    if (mode >> 32) {
      uint32_t permuted[32];
      const uint32_t *source = permuted;
      if (mode & GOC_DPP8)
        goc::dpp8_source(flags, exec_mask, mode, permuted, a[0]);
      else
        exec_mask = goc::dpp16_source(flags, exec_mask, mode, permuted, a[0]);
      return run<Bits>(flags, exec_mask, uint32_t(mode), d, &source, b);
    }
  }
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    *d = goc::class_x86_64_v4<Bits>(mode, a, b[0]) & exec_mask;
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    *d = goc::class_x86_64_v3<Bits>(mode, a, b[0]) & exec_mask;
    return GOC_SUCCESS;
  }
#endif
  constexpr unsigned fraction_bits = Bits == 16 ? 10 : Bits == 32 ? 23 : 20;
  constexpr uint32_t exponent_mask = Bits == 16 ? 31 : Bits == 32 ? 255 : 2047;
  constexpr uint32_t sign = Bits == 16 ? 0x8000 : 0x80000000;
  uint32_t result = 0;
  // Clang lowers vectorized variable shifts to FP conversions on baseline x86,
  // raising exception flags even with FENV_ACCESS enabled. Keep this path scalar.
#if defined(__clang__)
#pragma clang loop vectorize(disable)
#endif
  for (unsigned lane = 0; lane < 32; ++lane) {
    uint32_t high = a[Bits == 64 ? 1 : 0][lane], classes = b[0][lane];
    if constexpr (Bits == 16) {
      high = (high >> (mode & GOC_ALU_HIGH_A ? 16 : 0)) & 65535;
      classes >>= mode & GOC_ALU_HIGH_B ? 16 : 0;
    }
    if (mode & GOC_ALU_ABS_A)
      high &= ~sign;
    if (mode & GOC_ALU_NEG_A)
      high ^= sign;
    uint32_t exponent = (high >> fraction_bits) & exponent_mask;
    uint32_t fraction = high & ((1u << fraction_bits) - 1);
    if constexpr (Bits == 64)
      fraction |= a[0][lane];
    unsigned index;
    if (exponent == exponent_mask && fraction)
      index = (high >> (fraction_bits - 1)) & 1;
    else {
      index = exponent == exponent_mask ? 9 : exponent ? 8 : fraction ? 7 : 6;
      if (high & sign)
        index = 11 - index;
    }
    result |= ((classes >> index) & 1) << lane;
  }
  *d = result & exec_mask;
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_cmp_class_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  return run<16>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_class_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  return run<32>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmp_class_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  return run<64>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_class_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  return run<16>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_class_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  return run<32>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cmpx_class_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *d, const uint32_t *const *a, const uint32_t *const *b) {
  return run<64>(flags, exec_mask, instruction_flags, d, a, b);
}

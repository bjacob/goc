// SPDX-License-Identifier: MIT

#include "dot.h"
#include "goc/goc.h"
#include "internal.h"
#include "packed16.h"
#include "swmmac_float.h"

#include <array>
#include <stdint.h>

namespace {

template <bool Bf16, bool Packed, bool Wave64 = false>
int run(uint64_t flags, uint32_t mode, uint32_t *const *d, const uint32_t *const *a,
        const uint32_t *const *b, const uint32_t *const *index) {
  const uint32_t known = GOC_WMMA_NEG_LO_A | GOC_WMMA_NEG_LO_B | GOC_WMMA_NEG_HI_A |
                         GOC_WMMA_NEG_HI_B | GOC_SWMMAC_INDEX_KEY_1 |
                         (Wave64 ? GOC_SWMMAC_INDEX_KEY_2 : 0);
  if (int error = goc::validate(flags, mode & ~known, true))
    return error;
  // The shared kernels store 32-lane register sections. Packed Wave64
  // uses two physical VGPRs; FP32 Wave64 uses four.
  uint32_t *dest[Packed ? 4 : 8];
  for (unsigned reg = 0; reg < (Packed ? 4U : 8U); ++reg) {
    const unsigned count = Packed ? 2 : 4;
    dest[reg] = Wave64 ? d[reg % count] + 32 * (reg / count) : d[reg];
  }
  // Borrowed rocjitsu 184faac7d94: four DOT4 steps in compressed K order,
  // narrowing packed accumulators at each step. B negation follows compressed
  // pair position, irrespective of the selected dense element's parity.
  if ((flags & GOC_SEMANTICS_MASK) == GOC_SEMANTICS_EXACT_EMPIRICAL) {
    uint32_t result[Packed ? 4 : 8][32] = {};
    unsigned key = Wave64 ? 8 * ((mode >> 7) & 3) : (mode & GOC_SWMMAC_INDEX_KEY_1 ? 16 : 0);
    for (unsigned row = 0; row < 16; ++row)
      for (unsigned col = 0; col < 16; ++col) {
        unsigned reg = row % 8, lane = col + 16 * (row / 8);
        uint32_t acc = Packed ? uint16_t(dest[reg / 2][lane] >> (16 * (reg % 2))) : dest[reg][lane];
        for (unsigned stage = 0; stage < 4; ++stage) {
          std::array<uint16_t, 4> left, right;
          for (unsigned i = 0; i < 4; ++i) {
            unsigned ck = 4 * stage + i, group = ck / 2, slot = ck % 2;
            unsigned a_lane = Wave64 ? row + 16 * (ck / 4) : row + 16 * ((group / 2) & 1);
            unsigned a_reg = Wave64 ? (ck % 4) / 2 : 2 * (group / 4) + (group & 1);
            left[i] = uint16_t(a[a_reg][a_lane] >> (16 * slot));
            if (mode & (slot ? GOC_WMMA_NEG_HI_A : GOC_WMMA_NEG_LO_A))
              left[i] ^= 0x8000;
            unsigned local = Wave64 ? ck % 4 : 2 * (group & 1) + 4 * (group / 4) + slot;
            unsigned k = 4 * group + ((index[0][a_lane] >> (key + 2 * local)) & 3);
            unsigned b_lane = Wave64 ? col + 16 * (k / 8) : col + 16 * ((k / 8) & 1);
            unsigned b_slot = Wave64 ? k % 8 : 8 * (k / 16) + 2 * ((k / 2) & 3) + (k & 1);
            right[i] = uint16_t(b[b_slot / 2][b_lane] >> (16 * (b_slot % 2)));
            if (mode & (slot ? GOC_WMMA_NEG_HI_B : GOC_WMMA_NEG_LO_B))
              right[i] ^= 0x8000;
          }
          if constexpr (Packed)
            acc = goc::packed16::widen<Bf16>(uint16_t(acc));
          acc = goc::gfx12_dot_bits<Bf16, 4, Packed>(left, right, acc, flags & GOC_FP16_OVFL);
        }
        if constexpr (Packed)
          result[reg / 2][lane] |= acc << (16 * (reg % 2));
        else
          result[reg][lane] = acc;
      }
    // Read all input and tied C/D values before any potentially aliasing stores.
    for (unsigned reg = 0; reg < (Packed ? 4U : 8U); ++reg)
      for (unsigned lane = 0; lane < 32; ++lane)
        dest[reg][lane] = result[reg][lane];
    return GOC_SUCCESS;
  }
  goc::SwmmacFloatInputs input;
  goc::swmmac16_prepare<Bf16, Packed, Wave64>(input, mode, dest, a, b, index);
  bool saturate = (flags & GOC_FP16_OVFL) != 0;
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::swmmac_float_x86_64_v4<Bf16, Packed>(mode, dest, input, saturate);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::swmmac_float_x86_64_v3<Bf16, Packed>(mode, dest, input, saturate);
    return GOC_SUCCESS;
  }
#endif
  goc::swmmac_float_accumulate(mode, input);
  uint32_t result[Packed ? 4 : 8][32];
  goc::swmmac_float_pack<Bf16, Packed>(result, input.acc, saturate);
  for (unsigned reg = 0; reg < (Packed ? 4u : 8u); ++reg)
    for (unsigned lane = 0; lane < 32; ++lane)
      dest[reg][lane] = result[reg][lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_v_swmmac_f32_16x16x32_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                                  const uint32_t *const *a, const uint32_t *const *b,
                                  const uint32_t *const *index) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<false, false>(flags, instruction_flags, d, a, b, index);
}

int goc_v_swmmac_f32_16x16x32_bf16(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                                   const uint32_t *const *a, const uint32_t *const *b,
                                   const uint32_t *const *index) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<true, false>(flags, instruction_flags, d, a, b, index);
}

int goc_v_swmmac_f16_16x16x32_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                                  const uint32_t *const *a, const uint32_t *const *b,
                                  const uint32_t *const *index) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<false, true>(flags, instruction_flags, d, a, b, index);
}

int goc_v_swmmac_bf16_16x16x32_bf16(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                                    const uint32_t *const *a, const uint32_t *const *b,
                                    const uint32_t *const *index) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<true, true>(flags, instruction_flags, d, a, b, index);
}

int goc_v_swmmac_f32_16x16x32_f16_wave64(uint64_t flags, uint64_t instruction_flags,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b, const uint32_t *const *index) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<false, false, true>(flags, instruction_flags, d, a, b, index);
}

int goc_v_swmmac_f32_16x16x32_bf16_wave64(uint64_t flags, uint64_t instruction_flags,
                                          uint32_t *const *d, const uint32_t *const *a,
                                          const uint32_t *const *b, const uint32_t *const *index) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<true, false, true>(flags, instruction_flags, d, a, b, index);
}

int goc_v_swmmac_f16_16x16x32_f16_wave64(uint64_t flags, uint64_t instruction_flags,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b, const uint32_t *const *index) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<false, true, true>(flags, instruction_flags, d, a, b, index);
}

int goc_v_swmmac_bf16_16x16x32_bf16_wave64(uint64_t flags, uint64_t instruction_flags,
                                           uint32_t *const *d, const uint32_t *const *a,
                                           const uint32_t *const *b, const uint32_t *const *index) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<true, true, true>(flags, instruction_flags, d, a, b, index);
}

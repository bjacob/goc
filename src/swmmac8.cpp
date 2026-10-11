// SPDX-License-Identifier: MIT

#include "dot.h"
#include "float_formats.h"
#include "goc/goc.h"
#include "internal.h"
#include "swmmac_float.h"

#include <array>
#include <stdint.h>

namespace {

template <bool Bf8A, bool Bf8B, bool Wave64 = false>
int run(uint64_t flags, uint32_t mode, uint32_t *const *d, const uint32_t *const *a,
        const uint32_t *const *b, const uint32_t *const *index) {
  if (int error = goc::validate(
          flags, mode & ~(GOC_SWMMAC_INDEX_KEY_1 | (Wave64 ? GOC_SWMMAC_INDEX_KEY_2 : 0)), true))
    return error;
  // Existing accumulation kernels use eight 32-lane destination sections.
  // Wave64 stores the upper four sections in the upper lanes of D[0..3].
  uint32_t *dest[8];
  for (unsigned reg = 0; reg < 8; ++reg)
    dest[reg] = Wave64 ? d[reg % 4] + 32 * (reg / 4) : d[reg];
  unsigned key = Wave64 ? 8 * ((mode >> 7) & 3) : (mode & GOC_SWMMAC_INDEX_KEY_1 ? 16 : 0);
  // Compressed K remains contiguous for sparse DOT8 steps. Snapshot all
  // results before any stores because D can overlap A, B or the index VGPR.
  if ((flags & GOC_SEMANTICS_MASK) == GOC_SEMANTICS_EXACT_EMPIRICAL) {
    uint32_t result[8][32];
    for (unsigned row = 0; row < 16; ++row)
      for (unsigned col = 0; col < 16; ++col) {
        unsigned reg = row % 8, out_lane = col + 16 * (row / 8);
        uint32_t acc = dest[reg][out_lane];
        for (unsigned stage = 0; stage < 2; ++stage) {
          std::array<uint8_t, 8> left, right;
          for (unsigned i = 0; i < 8; ++i) {
            unsigned ck = stage * 8 + i;
            unsigned lane = row + 16 * (ck / 8) + (Wave64 ? 32 * ((ck / 4) % 2) : 0);
            unsigned slot = ck % (Wave64 ? 4 : 8);
            left[i] = uint8_t(a[slot / 4][lane] >> (8 * (slot % 4)));
            unsigned k = 4 * (ck / 2) + ((index[0][lane] >> (key + 2 * slot)) & 3);
            unsigned b_lane = col + 16 * (k / 16) + (Wave64 ? 32 * ((k / 8) % 2) : 0);
            unsigned b_reg = (k % (Wave64 ? 8 : 16)) / 4;
            right[i] = uint8_t(b[b_reg][b_lane] >> (8 * (k % 4)));
          }
          acc = goc::gfx12_dot8_f32<Bf8A, Bf8B>(left, right, acc);
        }
        result[reg][out_lane] = acc;
      }
    for (unsigned reg = 0; reg < 8; ++reg)
      for (unsigned lane = 0; lane < 32; ++lane)
        dest[reg][lane] = result[reg][lane];
    return GOC_SUCCESS;
  }
  goc::SwmmacFloatInputs input;
  for (unsigned row = 0; row < 16; ++row)
    for (unsigned ck = 0; ck < 16; ++ck) {
      unsigned lane = row + 16 * (ck / 8) + (Wave64 ? 32 * ((ck / 4) % 2) : 0);
      unsigned slot = ck % (Wave64 ? 4 : 8);
      input.a[row][ck] = goc::fp8_to_float<Bf8A>(uint8_t(a[slot / 4][lane] >> (8 * (slot % 4))));
      input.selected[row][ck] = uint8_t(4 * (ck / 2) + ((index[0][lane] >> (key + 2 * slot)) & 3));
    }
  for (unsigned k = 0; k < 32; ++k)
    for (unsigned col = 0; col < 16; ++col) {
      unsigned lane = col + 16 * (k / 16) + (Wave64 ? 32 * ((k / 8) % 2) : 0);
      unsigned reg = (k % (Wave64 ? 8 : 16)) / 4;
      input.b[k][col] = goc::fp8_to_float<Bf8B>(uint8_t(b[reg][lane] >> (8 * (k % 4))));
    }
  for (unsigned row = 0; row < 16; ++row)
    for (unsigned col = 0; col < 16; ++col)
      input.acc[row][col] = goc::as_float(dest[row % 8][col + 16 * (row / 8)]);
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::swmmac_float_x86_64_v4<false, false>(0, dest, input, false);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::swmmac_float_x86_64_v3<false, false>(0, dest, input, false);
    return GOC_SUCCESS;
  }
#endif
  goc::swmmac_float_accumulate(0, input);
  uint32_t result[8][32];
  goc::swmmac_float_pack<false, false>(result, input.acc, false);
  for (unsigned reg = 0; reg < 8; ++reg)
    for (unsigned lane = 0; lane < 32; ++lane)
      dest[reg][lane] = result[reg][lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_v_swmmac_f32_16x16x32_fp8_fp8(uint64_t flags, uint64_t instruction_flags,
                                      uint32_t *const *d, const uint32_t *const *a,
                                      const uint32_t *const *b, const uint32_t *const *index) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<false, false>(flags, instruction_flags, d, a, b, index);
}

int goc_v_swmmac_f32_16x16x32_fp8_bf8(uint64_t flags, uint64_t instruction_flags,
                                      uint32_t *const *d, const uint32_t *const *a,
                                      const uint32_t *const *b, const uint32_t *const *index) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<false, true>(flags, instruction_flags, d, a, b, index);
}

int goc_v_swmmac_f32_16x16x32_bf8_fp8(uint64_t flags, uint64_t instruction_flags,
                                      uint32_t *const *d, const uint32_t *const *a,
                                      const uint32_t *const *b, const uint32_t *const *index) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<true, false>(flags, instruction_flags, d, a, b, index);
}

int goc_v_swmmac_f32_16x16x32_bf8_bf8(uint64_t flags, uint64_t instruction_flags,
                                      uint32_t *const *d, const uint32_t *const *a,
                                      const uint32_t *const *b, const uint32_t *const *index) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<true, true>(flags, instruction_flags, d, a, b, index);
}

int goc_v_swmmac_f32_16x16x32_fp8_fp8_wave64(uint64_t flags, uint64_t instruction_flags,
                                             uint32_t *const *d, const uint32_t *const *a,
                                             const uint32_t *const *b,
                                             const uint32_t *const *index) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<false, false, true>(flags, instruction_flags, d, a, b, index);
}

int goc_v_swmmac_f32_16x16x32_fp8_bf8_wave64(uint64_t flags, uint64_t instruction_flags,
                                             uint32_t *const *d, const uint32_t *const *a,
                                             const uint32_t *const *b,
                                             const uint32_t *const *index) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<false, true, true>(flags, instruction_flags, d, a, b, index);
}

int goc_v_swmmac_f32_16x16x32_bf8_fp8_wave64(uint64_t flags, uint64_t instruction_flags,
                                             uint32_t *const *d, const uint32_t *const *a,
                                             const uint32_t *const *b,
                                             const uint32_t *const *index) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<true, false, true>(flags, instruction_flags, d, a, b, index);
}

int goc_v_swmmac_f32_16x16x32_bf8_bf8_wave64(uint64_t flags, uint64_t instruction_flags,
                                             uint32_t *const *d, const uint32_t *const *a,
                                             const uint32_t *const *b,
                                             const uint32_t *const *index) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<true, true, true>(flags, instruction_flags, d, a, b, index);
}

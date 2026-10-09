// SPDX-License-Identifier: MIT

#include "float_formats.h"
#include "goc/goc.h"
#include "internal.h"
#include "swmmac_float.h"

#include <stdint.h>

namespace {

template <bool Bf8A, bool Bf8B>
int run(uint64_t flags, uint32_t mode, uint32_t *const *d, const uint32_t *const *a,
        const uint32_t *const *b, const uint32_t *const *index) {
  if (int error = goc::validate(flags, mode & ~GOC_SWMMAC_INDEX_KEY_1))
    return error;
  goc::SwmmacFloatInputs input;
  unsigned key = mode & GOC_SWMMAC_INDEX_KEY_1 ? 16 : 0;
  for (unsigned row = 0; row < 16; ++row)
    for (unsigned ck = 0; ck < 16; ++ck) {
      unsigned lane = row + 16 * (ck / 8), slot = ck % 8;
      input.a[row][ck] = goc::fp8_to_float<Bf8A>(uint8_t(a[slot / 4][lane] >> (8 * (slot % 4))));
      input.selected[row][ck] = uint8_t(4 * (ck / 2) + ((index[0][lane] >> (key + 2 * slot)) & 3));
    }
  for (unsigned k = 0; k < 32; ++k)
    for (unsigned col = 0; col < 16; ++col)
      input.b[k][col] =
          goc::fp8_to_float<Bf8B>(uint8_t(b[(k % 16) / 4][col + 16 * (k / 16)] >> (8 * (k % 4))));
  for (unsigned row = 0; row < 16; ++row)
    for (unsigned col = 0; col < 16; ++col)
      input.acc[row][col] = goc::as_float(d[row % 8][col + 16 * (row / 8)]);
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::swmmac_float_x86_64_v4<false, false>(0, d, input, false);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::swmmac_float_x86_64_v3<false, false>(0, d, input, false);
    return GOC_SUCCESS;
  }
#endif
  goc::swmmac_float_accumulate(0, input);
  uint32_t result[8][32];
  goc::swmmac_float_pack<false, false>(result, input.acc, false);
  for (unsigned reg = 0; reg < 8; ++reg)
    for (unsigned lane = 0; lane < 32; ++lane)
      d[reg][lane] = result[reg][lane];
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

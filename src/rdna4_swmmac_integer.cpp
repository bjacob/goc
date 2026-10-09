// SPDX-License-Identifier: MIT

#include "rdna4_swmmac_integer.h"
#include "goc/goc.h"
#include "internal.h"

#include <algorithm>
#include <stdint.h>

namespace {

template <unsigned Bits, unsigned K>
int run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *index) {
  const uint32_t known = GOC_WMMA_SIGNED_A | GOC_WMMA_SIGNED_B | GOC_WMMA_CLAMP |
                         (K == 32 ? GOC_SWMMAC_INDEX_KEY_1 : 0);
  if (int error = goc::validate(flags, mode & ~known, true))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
  goc::SwmmacIntegerInputs input;
  goc::swmmac_integer_prepare<Bits, K>(input, mode, d, a, b, index);
  bool clamp = mode & GOC_WMMA_CLAMP;
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::swmmac_integer_x86_64_v4(K, exec_mask, clamp, d, input);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::swmmac_integer_x86_64_v3(K, exec_mask, clamp, d, input);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[8][32];
  for (unsigned row = 0; row < 16; ++row)
    for (unsigned col = 0; col < 16; ++col) {
      int64_t acc = input.acc[row][col];
      for (unsigned stage = 0; stage < 2; ++stage) {
        for (unsigned local = 0; local < K / 4; ++local) {
          unsigned ck = goc::swmmac_integer_ck(stage, local);
          acc += int64_t(input.a[row][ck]) * input.b[input.selected[row][ck]][col];
        }
        if (clamp)
          acc = std::clamp(acc, -INT64_C(2147483648), INT64_C(2147483647));
      }
      result[row % 8][col + 16 * (row / 8)] = uint32_t(acc);
    }
  for (unsigned reg = 0; reg < 8; ++reg)
    for (unsigned lane = 0; lane < 32; ++lane)
      if ((exec_mask >> lane) & 1)
        d[reg][lane] = result[reg][lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_swmmac_i32_16x16x32_iu8(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *index) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<8, 32>(flags, exec_mask, instruction_flags, d, a, b, index);
}

int goc_rdna4_v_swmmac_i32_16x16x32_iu4(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *index) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<4, 32>(flags, exec_mask, instruction_flags, d, a, b, index);
}

int goc_rdna4_v_swmmac_i32_16x16x64_iu4(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *index) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<4, 64>(flags, exec_mask, instruction_flags, d, a, b, index);
}

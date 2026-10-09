// SPDX-License-Identifier: MIT

// Lane selection and inactive-source rules follow rocjitsu's RDNA4 VOP3 models.

#include "permlane.h"
#include "goc/goc.h"
#include "internal.h"

#include <stdint.h>

namespace {

template <bool Cross, bool Var>
int run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, uint32_t lo, uint32_t hi) {
  if (int error = goc::validate(flags, mode & ~(GOC_PERMLANE_FI | GOC_PERMLANE_BOUND_CTRL), true,
                                GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
  const uint32_t *indices = Var ? b[0] : nullptr;
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::permlane_x86_64_v4<Cross, Var>(exec_mask, mode, d[0], a[0], indices, lo, hi);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::permlane_x86_64_v3<Cross, Var>(exec_mask, mode, d[0], a[0], indices, lo, hi);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (unsigned lane = 0; lane < 32; ++lane) {
    unsigned sel = Var ? indices[lane] : ((lane & 8 ? hi : lo) >> ((lane & 7) * 4));
    unsigned source = ((lane & 16) ^ (Cross ? 16 : 0)) | (sel & 15);
    bool readable = ((exec_mask >> source) & 1) || (mode & GOC_PERMLANE_FI);
    result[lane] = readable ? a[0][source] : (mode & GOC_PERMLANE_BOUND_CTRL ? 0 : d[0][lane]);
  }
  for (unsigned lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_v_permlane16_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, uint32_t lo, uint32_t hi) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<false, false>(flags, exec_mask, instruction_flags, d, a, nullptr, lo, hi);
}

int goc_v_permlanex16_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                          uint32_t *const *d, const uint32_t *const *a, uint32_t lo, uint32_t hi) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<true, false>(flags, exec_mask, instruction_flags, d, a, nullptr, lo, hi);
}

int goc_v_permlane16_var_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a,
                             const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<false, true>(flags, exec_mask, instruction_flags, d, a, b, 0, 0);
}

int goc_v_permlanex16_var_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<true, true>(flags, exec_mask, instruction_flags, d, a, b, 0, 0);
}

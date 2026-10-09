// SPDX-License-Identifier: MIT

#include "rdna4_cube.h"
#include "goc/goc.h"
#include "internal.h"

#include <stdint.h>

namespace {

template <goc::Cube Op>
int execute(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (int error = goc::validate(flags, mode & ~UINT32_C(0x1ff), true))
    return error;
  if (!uint32_t(mask))
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::cube_x86_64_v4<Op>(uint32_t(mask), mode, d[0], a[0], b[0], c[0]);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::cube_x86_64_v3<Op>(uint32_t(mask), mode, d[0], a[0], b[0], c[0]);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (unsigned lane = 0; lane < 32; ++lane)
    result[lane] = goc::cube_value<Op>(a[0][lane], b[0][lane], c[0][lane], mode);
  for (unsigned lane = 0; lane < 32; ++lane)
    if ((mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_cubeid_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return execute<goc::Cube::Id>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_rdna4_v_cubesc_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return execute<goc::Cube::Sc>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_rdna4_v_cubetc_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return execute<goc::Cube::Tc>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_rdna4_v_cubema_f32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return execute<goc::Cube::Ma>(flags, exec_mask, instruction_flags, d, a, b, c);
}

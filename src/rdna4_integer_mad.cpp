// SPDX-License-Identifier: MIT
// Wide intermediates and integer saturation follow rocjitsu shared/simd_glue.h.

#include "rdna4_integer_mad.h"
#include "goc/goc.h"
#include "internal.h"

#include <algorithm>
#include <stdint.h>

namespace {

template <int Bits, bool Signed> int64_t input(uint32_t bits) {
  bits &= uint32_t((UINT64_C(1) << Bits) - 1);
  return int64_t(bits) - (Signed && (bits & (UINT32_C(1) << (Bits - 1))) ? INT64_C(1) << Bits : 0);
}

template <int Bits, bool Signed>
int mad(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d, const uint32_t *const *a,
        const uint32_t *const *b, const uint32_t *const *c) {
  const uint32_t known = GOC_ALU_CLAMP | (Bits == 16 ? GOC_ALU_HIGH_A | GOC_ALU_HIGH_B : 0);
  if (int error = goc::validate(flags, mode & ~known))
    return error;
  if (uint32_t(mask) == 0)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::integer_mad_x86_64_v3<Bits, Signed>(uint32_t(mask), mode, d[0], a[0], b[0], c[0]);
    return GOC_SUCCESS;
  }
#endif
  const int sa = Bits == 16 && (mode & GOC_ALU_HIGH_A) ? 16 : 0;
  const int sb = Bits == 16 && (mode & GOC_ALU_HIGH_B) ? 16 : 0;
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    int64_t value = input<Bits, Signed>(a[0][lane] >> sa) * input<Bits, Signed>(b[0][lane] >> sb) +
                    input<32, Signed>(c[0][lane]);
    if (mode & GOC_ALU_CLAMP)
      value = std::clamp(value, Signed ? int64_t(INT32_MIN) : INT64_C(0),
                         Signed ? int64_t(INT32_MAX) : int64_t(UINT32_MAX));
    result[lane] = uint32_t(value);
  }
  for (int lane = 0; lane < 32; ++lane)
    if (mask >> lane & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_mad_u32_u16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return mad<16, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_rdna4_v_mad_i32_i16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return mad<16, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_rdna4_v_mad_u32_u24(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return mad<24, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_rdna4_v_mad_i32_i24(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return mad<24, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

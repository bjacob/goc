// SPDX-License-Identifier: MIT
// Integer DOT semantics follow rocjitsu codegen/execute/packed.py.

#include "goc/goc.h"
#include "internal.h"
#include "rdna4_simd.h"

#include <algorithm>
#include <stdint.h>

namespace {

template <int Bits, bool Unsigned>
int dot(uint64_t flags, uint64_t mask, uint32_t modifiers, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  const uint32_t known = GOC_DOT_CLAMP | (Unsigned ? 0 : GOC_DOT_SIGNED_A | GOC_DOT_SIGNED_B);
  if (int error = goc::validate(flags, modifiers & ~known, true))
    return error;
  if (uint32_t(mask) == 0)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::integer_dot_x86_64_v3(Bits, Unsigned, uint32_t(mask), modifiers, d[0], a[0], b[0], c[0]);
    return GOC_SUCCESS;
  }
#endif
  const int sign_a = modifiers & GOC_DOT_SIGNED_A ? 1 << (Bits - 1) : 0;
  const int sign_b = modifiers & GOC_DOT_SIGNED_B ? 1 << (Bits - 1) : 0;
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    int64_t sum = c[0][lane];
    if (!Unsigned && (c[0][lane] & 0x80000000))
      sum -= INT64_C(1) << 32;
    for (int shift = 0; shift < 32; shift += Bits) {
      int x = ((a[0][lane] >> shift) & ((1 << Bits) - 1));
      int y = ((b[0][lane] >> shift) & ((1 << Bits) - 1));
      sum += ((x ^ sign_a) - sign_a) * ((y ^ sign_b) - sign_b);
    }
    if (modifiers & GOC_DOT_CLAMP)
      sum = std::clamp(sum, Unsigned ? INT64_C(0) : int64_t(INT32_MIN),
                       Unsigned ? int64_t(UINT32_MAX) : int64_t(INT32_MAX));
    result[lane] = uint32_t(sum);
  }
  for (int lane = 0; lane < 32; ++lane)
    if ((mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_dot4_i32_iu8(uint64_t flags, uint64_t mask, uint64_t modifiers, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c) {
  if (modifiers >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return dot<8, false>(flags, mask, modifiers, d, a, b, c);
}

int goc_rdna4_v_dot4_u32_u8(uint64_t flags, uint64_t mask, uint64_t modifiers, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  if (modifiers >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return dot<8, true>(flags, mask, modifiers, d, a, b, c);
}

int goc_rdna4_v_dot8_i32_iu4(uint64_t flags, uint64_t mask, uint64_t modifiers, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c) {
  if (modifiers >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return dot<4, false>(flags, mask, modifiers, d, a, b, c);
}

int goc_rdna4_v_dot8_u32_u4(uint64_t flags, uint64_t mask, uint64_t modifiers, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  if (modifiers >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return dot<4, true>(flags, mask, modifiers, d, a, b, c);
}

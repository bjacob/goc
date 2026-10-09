// SPDX-License-Identifier: MIT
// SAD and saturation rules adapted from rocjitsu shared/simd_glue.h and
// lib/python/amdisa/codegen/execute/vector_special.py.

#include "rdna4_sad.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_dpp.h"

#include <algorithm>
#include <stdint.h>

namespace {

template <int Bits, bool Masked> uint32_t difference(uint32_t a, uint32_t b) {
  const uint32_t field_mask = UINT32_MAX >> (32 - Bits);
  uint32_t sum = 0;
  for (int field = 0; field < 32 / Bits; ++field) {
    uint32_t x = (a >> (field * Bits)) & field_mask;
    uint32_t y = (b >> (field * Bits)) & field_mask;
    if (!Masked || y)
      sum += x > y ? x - y : y - x;
  }
  return sum;
}

template <goc::Sad Op>
int sad(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d, const uint32_t *const *a,
        const uint32_t *const *b, const uint32_t *const *c) {
  if (mode >> 32) {
    if constexpr (goc::sad_quad(Op))
      return GOC_ERROR_INVALID_FLAGS;
    else
      return goc::execute_dpp(flags, mask, mode, a,
                              [&](uint32_t effective, const uint32_t *const *source) {
                                return sad<Op>(flags, effective, uint32_t(mode), d, source, b, c);
                              });
  }
  if (int error = goc::validate(flags, mode & ~GOC_ALU_CLAMP))
    return error;
  if (!mask)
    return GOC_SUCCESS;
  const bool clamp = mode & GOC_ALU_CLAMP;
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::sad_x86_64_v3<Op>(mask, clamp, d, a, b[0], c);
    return GOC_SUCCESS;
  }
#endif
  constexpr int outputs = goc::sad_outputs(Op);
  constexpr int bits = Op == goc::Sad::U32 ? 32 : Op == goc::Sad::U16 ? 16 : 8;
  constexpr int windows = goc::sad_quad(Op) ? 4 : 1;
  uint32_t result[outputs][32] = {};
  for (int lane = 0; lane < 32; ++lane) {
    uint64_t source = a[0][lane];
    if constexpr (goc::sad_quad(Op))
      source |= uint64_t(a[1][lane]) << 32;
    for (int window = 0; window < windows; ++window) {
      uint32_t sum =
          difference<bits, goc::sad_masked(Op)>(uint32_t(source >> (8 * window)), b[0][lane]);
      if constexpr (Op == goc::Sad::HighU8)
        sum <<= 16;
      if constexpr (goc::sad_packed(Op)) {
        const unsigned shift = 16 * (window % 2);
        uint32_t value = sum + ((c[window / 2][lane] >> shift) & 0xffff);
        value = clamp ? std::min(value, 0xffffU) : value & 0xffff;
        result[window / 2][lane] |= value << shift;
      } else {
        uint64_t value = uint64_t(sum) + c[window][lane];
        result[window][lane] = uint32_t(clamp ? std::min(value, uint64_t(UINT32_MAX)) : value);
      }
    }
  }
  for (int reg = 0; reg < outputs; ++reg)
    for (int lane = 0; lane < 32; ++lane)
      if ((mask >> lane) & 1)
        d[reg][lane] = result[reg][lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_sad_u8(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
                       const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *c) {
  return sad<goc::Sad::U8>(flags, mask, mode, d, a, b, c);
}

int goc_rdna4_v_sad_hi_u8(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *c) {
  return sad<goc::Sad::HighU8>(flags, mask, mode, d, a, b, c);
}

int goc_rdna4_v_sad_u16(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c) {
  return sad<goc::Sad::U16>(flags, mask, mode, d, a, b, c);
}

int goc_rdna4_v_sad_u32(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c) {
  return sad<goc::Sad::U32>(flags, mask, mode, d, a, b, c);
}

int goc_rdna4_v_msad_u8(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c) {
  return sad<goc::Sad::MaskedU8>(flags, mask, mode, d, a, b, c);
}

int goc_rdna4_v_qsad_pk_u16_u8(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
                               const uint32_t *const *a, const uint32_t *const *b,
                               const uint32_t *const *c) {
  return sad<goc::Sad::QuadU16>(flags, mask, mode, d, a, b, c);
}

int goc_rdna4_v_mqsad_pk_u16_u8(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                const uint32_t *const *c) {
  return sad<goc::Sad::MaskedQuadU16>(flags, mask, mode, d, a, b, c);
}

int goc_rdna4_v_mqsad_u32_u8(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c) {
  return sad<goc::Sad::MaskedQuadU32>(flags, mask, mode, d, a, b, c);
}

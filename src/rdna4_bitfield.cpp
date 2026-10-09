// SPDX-License-Identifier: MIT

#include "rdna4_bitfield.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_dpp.h"

#include <stdint.h>

namespace {

template <goc::Bitfield Op> uint32_t evaluate(uint32_t a, uint32_t b, uint32_t c) {
  if constexpr (Op == goc::Bitfield::ExtractUnsigned || Op == goc::Bitfield::ExtractSigned) {
    unsigned offset = b & 31, width = c & 31;
    uint32_t field_mask = (uint32_t(1) << width) - 1;
    if constexpr (Op == goc::Bitfield::ExtractUnsigned)
      return (a >> offset) & field_mask;
    // rocjitsu's simd_bfe_i32 sign-extends A before extracting: fields
    // crossing bit 31 include copies of A's sign bit.
    uint32_t sign = 0u - (a >> 31);
    uint32_t field = (((a ^ sign) >> offset) ^ sign) & field_mask;
    uint32_t sign_bit = (uint32_t(1) << width) >> 1;
    return (field ^ sign_bit) - sign_bit;
  }
  if constexpr (Op == goc::Bitfield::AlignBit || Op == goc::Bitfield::AlignByte) {
    unsigned shift = Op == goc::Bitfield::AlignBit ? c & 31 : (c & 3) * 8;
    return uint32_t(((uint64_t(a) << 32) | b) >> shift);
  }
  if constexpr (Op == goc::Bitfield::Permute) {
    // Byte selection and sign-fill semantics follow rocjitsu's perm_b32.
    uint64_t source = (uint64_t(a) << 32) | b;
    uint32_t result = 0;
    for (unsigned byte = 0; byte < 4; ++byte) {
      unsigned selector = (c >> (8 * byte)) & 255;
      uint32_t value;
      if (selector < 8)
        value = uint32_t(source >> (8 * selector)) & 255;
      else if (selector < 12)
        value = ((source >> (16 * (selector - 8) + 15)) & 1) ? 255 : 0;
      else
        value = selector == 12 ? 0 : 255;
      result |= value << (8 * byte);
    }
    return result;
  }
  if constexpr (Op == goc::Bitfield::Insert)
    return (a & b) | (~a & c);
  if constexpr (Op == goc::Bitfield::Mask)
    return ((uint32_t(1) << (a & 31)) - 1) << (b & 31);
  if constexpr (Op == goc::Bitfield::Reverse) {
    a = ((a & 0x55555555) << 1) | ((a >> 1) & 0x55555555);
    a = ((a & 0x33333333) << 2) | ((a >> 2) & 0x33333333);
    a = ((a & 0x0f0f0f0f) << 4) | ((a >> 4) & 0x0f0f0f0f);
    a = ((a & 0x00ff00ff) << 8) | ((a >> 8) & 0x00ff00ff);
    return (a << 16) | (a >> 16);
  }
}

template <goc::Bitfield Op>
int bitfield(uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
             const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (mode >> 32)
    return goc::execute_dpp(
        flags, mask, mode, a, [&](uint32_t effective, const uint32_t *const *source) {
          return bitfield<Op>(flags, effective, uint32_t(mode), d, source, b, c);
        });

  if (int error = goc::validate(flags, mode))
    return error;
  if (!uint32_t(mask))
    return GOC_SUCCESS;
  const uint32_t *bp = nullptr, *cp = nullptr;
  if constexpr (Op != goc::Bitfield::Reverse)
    bp = b[0];
  if constexpr (Op != goc::Bitfield::Reverse && Op != goc::Bitfield::Mask)
    cp = c[0];
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::bitfield_x86_64_v4<Op>(uint32_t(mask), d[0], a[0], bp, cp);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  // BFI's AVX2 candidate provided no substantial gain over baseline.
  if constexpr (Op != goc::Bitfield::Insert) {
    if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
      goc::bitfield_x86_64_v3<Op>(uint32_t(mask), d[0], a[0], bp, cp);
      return GOC_SUCCESS;
    }
  }
#endif
  uint32_t result[32];
  if constexpr (Op == goc::Bitfield::ExtractUnsigned || Op == goc::Bitfield::ExtractSigned) {
    // Clang's baseline vector shifts use FP conversions for powers of two;
    // converting 2^31 raises FE_INVALID. Prevent both loop and SLP vectorization.
#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
#endif
    for (int lane = 0; lane < 32; ++lane)
      result[lane] = evaluate<Op>(a[0][lane], bp[lane], cp[lane]);
  } else {
    for (int lane = 0; lane < 32; ++lane)
      result[lane] = evaluate<Op>(a[0][lane], bp ? bp[lane] : 0, cp ? cp[lane] : 0);
  }
  for (int lane = 0; lane < 32; ++lane)
    if ((mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_bfe_u32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c) {
  return bitfield<goc::Bitfield::ExtractUnsigned>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_rdna4_v_bfe_i32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c) {
  return bitfield<goc::Bitfield::ExtractSigned>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_rdna4_v_bfi_b32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c) {
  return bitfield<goc::Bitfield::Insert>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_rdna4_v_bfm_b32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  return bitfield<goc::Bitfield::Mask>(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

int goc_rdna4_v_bfrev_b32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                          uint32_t *const *d, const uint32_t *const *a) {
  return bitfield<goc::Bitfield::Reverse>(flags, exec_mask, instruction_flags, d, a, nullptr,
                                          nullptr);
}

int goc_rdna4_v_alignbit_b32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c) {
  return bitfield<goc::Bitfield::AlignBit>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_rdna4_v_alignbyte_b32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c) {
  return bitfield<goc::Bitfield::AlignByte>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_rdna4_v_perm_b32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c) {
  return bitfield<goc::Bitfield::Permute>(flags, exec_mask, instruction_flags, d, a, b, c);
}

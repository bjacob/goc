// SPDX-License-Identifier: MIT

#include "rdna4_integer_conversion.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_dpp.h"

#include <stdint.h>

namespace {

template <bool Unsigned> uint32_t narrow(uint32_t raw) {
  // Saturation follows rocjitsu's execute_v_cvt_pk_[iu]16_[iu]32 handlers.
  if constexpr (Unsigned)
    return raw > 65535 ? 65535 : raw;
  else {
    // Adding 32768 maps the entire signed 16-bit interval to [0,65535].
    // Unsigned wraparound makes the range test valid for all 32-bit inputs.
    return raw + 32768u > 65535 ? (raw >> 31 ? 0x8000 : 0x7fff) : raw & 65535;
  }
}

template <bool Unsigned, bool Packed>
int convert(uint64_t flags, uint32_t mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b) {
  if (mode >> 32)
    return goc::execute_dpp(
        flags, mask, mode, a, [&](uint32_t effective, const uint32_t *const *source) {
          return convert<Unsigned, Packed>(flags, effective, uint32_t(mode), d, source, b);
        });
  if (int error = goc::validate(flags, mode & ~(Packed ? 0 : GOC_ALU_HIGH_A)))
    return error;
  if (!mask)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::integer_conversion_x86_64_v4<Unsigned, Packed>(mask, mode, d[0], a[0],
                                                        Packed ? b[0] : nullptr);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if constexpr (Packed) {
    if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
      goc::integer_conversion_x86_64_v3<Unsigned>(mask, d[0], a[0], b[0]);
      return GOC_SUCCESS;
    }
  }
#endif
  uint32_t result[32];
  for (unsigned lane = 0; lane < 32; ++lane) {
    if constexpr (Packed)
      result[lane] = narrow<Unsigned>(a[0][lane]) | (narrow<Unsigned>(b[0][lane]) << 16);
    else {
      uint32_t half = (a[0][lane] >> (mode & GOC_ALU_HIGH_A ? 16 : 0)) & 65535;
      result[lane] = Unsigned ? half : (half ^ 0x8000) - 0x8000;
    }
  }
  for (unsigned lane = 0; lane < 32; ++lane)
    if ((mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_cvt_i32_i16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a) {
  return convert<false, false>(flags, exec_mask, instruction_flags, d, a, nullptr);
}

int goc_rdna4_v_cvt_u32_u16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a) {
  return convert<true, false>(flags, exec_mask, instruction_flags, d, a, nullptr);
}

int goc_rdna4_v_cvt_pk_i16_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b) {
  return convert<false, true>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_cvt_pk_u16_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b) {
  return convert<true, true>(flags, exec_mask, instruction_flags, d, a, b);
}

// SPDX-License-Identifier: MIT

#include "dot.h"
#include "dpp.h"
#include "float_formats.h"
#include "gfx11_dot2.h"
#include "goc/goc.h"
#include "internal.h"
#include "simd.h"

#include <array>
#include <cmath>
#include <stdint.h>

namespace {

template <bool Bf16, bool Rdna4>
int dot(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (instruction_flags >> 32)
    return goc::execute_dpp(flags, exec_mask, instruction_flags, a,
                            [&](uint32_t exec_mask, const uint32_t *const *source) {
                              return dot<Bf16, Rdna4>(flags, exec_mask, uint32_t(instruction_flags),
                                                      d, source, b, c);
                            });
  // Bits 0..4: negation; bit 6: CLAMP; bits 7..10: half selection.
  if (int error = goc::validate(flags, instruction_flags & ~0x7dfU, true))
    return error;
  if (exec_mask == 0)
    return GOC_SUCCESS;

#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_EXACT_EMPIRICAL &&
      (flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::dot2_x86_64_v3(Bf16, exec_mask, instruction_flags, d[0], a[0], b[0], c[0]);
    return GOC_SUCCESS;
  }
#endif
  const int a0_shift = instruction_flags & GOC_DOT_LO_A_HIGH ? 16 : 0;
  const int b0_shift = instruction_flags & GOC_DOT_LO_B_HIGH ? 16 : 0;
  const int a1_shift = instruction_flags & GOC_DOT_HI_A_LOW ? 0 : 16;
  const int b1_shift = instruction_flags & GOC_DOT_HI_B_LOW ? 0 : 16;
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    std::array<uint16_t, 2> left = {uint16_t(a[0][lane] >> a0_shift),
                                    uint16_t(a[0][lane] >> a1_shift)};
    std::array<uint16_t, 2> right = {uint16_t(b[0][lane] >> b0_shift),
                                     uint16_t(b[0][lane] >> b1_shift)};
    if (instruction_flags & GOC_DOT_NEG_LO_A)
      left[0] ^= 0x8000;
    if (instruction_flags & GOC_DOT_NEG_HI_A)
      left[1] ^= 0x8000;
    if (instruction_flags & GOC_DOT_NEG_LO_B)
      right[0] ^= 0x8000;
    if (instruction_flags & GOC_DOT_NEG_HI_B)
      right[1] ^= 0x8000;
    uint32_t acc_bits = c[0][lane] ^ (instruction_flags & GOC_DOT_NEG_C ? 0x80000000 : 0);
    if ((flags & GOC_SEMANTICS_MASK) == GOC_SEMANTICS_EXACT_EMPIRICAL) {
      if constexpr (Rdna4)
        result[lane] = goc::gfx12_dot_bits<Bf16, 2>(left, right, acc_bits);
      else
        result[lane] = goc::gfx11_dot2_f32<Bf16>(left[0], right[0], left[1], right[1], acc_bits);
    } else {
      float acc = goc::as_float(acc_bits);
      for (int j = 0; j < 2; ++j) {
        float x = Bf16 ? goc::bf16_to_float(left[j]) : goc::f16_to_float(left[j]);
        float y = Bf16 ? goc::bf16_to_float(right[j]) : goc::f16_to_float(right[j]);
        acc = std::fma(x, y, acc);
      }
      result[lane] = goc::as_bits(acc);
    }
  }

  for (int lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

// RX 9070 DOT2 does not generate EXCP_FLAG_USER updates, including NaNs,
// denormals and overflow. Reporting leaves the register unchanged.
int goc_v_dot2_f32_f16_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return dot<false, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_v_dot2_f32_bf16_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c,
                              uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) > GOC_SEMANTICS_EXACT_EMPIRICAL)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return dot<true, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_v_dot2_f32_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *c, uint32_t *excp_flag_user) {
  // GFX11 result bits are characterized; its exception-register updates are not.
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return dot<false, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_v_dot2_f32_bf16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c, uint32_t *excp_flag_user) {
  // GFX11 result bits are characterized; its exception-register updates are not.
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return dot<true, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

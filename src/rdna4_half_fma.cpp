// SPDX-License-Identifier: MIT

#include "rdna4_half_fma.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_dpp.h"
#include "rdna4_fma.h"
#include "rdna4_half_fma_scalar.h"

#include <stdint.h>

namespace {

template <goc::FmaOperands Operands = goc::FmaOperands::Registers>
int run(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
        uint16_t literal = 0) {
  if (mode >> 32) {
    if constexpr (Operands == goc::FmaOperands::Registers) {
      return goc::execute_dpp(
          flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
            return run<Operands>(flags, exec_mask, uint32_t(mode), d, source, b, c, literal);
          });
    } else {
      return GOC_ERROR_INVALID_FLAGS;
    }
  }
  const uint32_t known = Operands == goc::FmaOperands::Registers
                             ? 0x1fffU
                             : GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_D;
  if (int error = goc::validate(flags, mode & ~known, true))
    return error;
  if (exec_mask == 0)
    return GOC_SUCCESS;
#if defined(GOC_HAVE_X86_64_V3)
  bool exact = (flags & GOC_SEMANTICS_MASK) == GOC_SEMANTICS_EXACT_EMPIRICAL;
  if (!exact && (flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::half_fma_x86_64_v3<Operands>(
        bool(flags & GOC_FP16_OVFL), exec_mask, mode, d[0], a[0], b[0],
        Operands == goc::FmaOperands::Registers ? c[0] : nullptr, literal);
    return GOC_SUCCESS;
  }
#endif
  int a_shift = mode & GOC_ALU_HIGH_A ? 16 : 0;
  int b_shift = mode & GOC_ALU_HIGH_B ? 16 : 0;
  int c_shift = mode & GOC_ALU_HIGH_C ? 16 : 0;
  int d_shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  uint16_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    uint16_t x = uint16_t(a[0][lane] >> a_shift), y = uint16_t(b[0][lane] >> b_shift), z;
    if constexpr (Operands == goc::FmaOperands::MultiplyLiteral) {
      z = y;
      y = literal;
    } else if constexpr (Operands == goc::FmaOperands::AddLiteral) {
      z = literal;
    } else {
      z = uint16_t(c[0][lane] >> c_shift);
    }
    result[lane] = goc::half_fma_value(x, y, z, mode, flags & GOC_FP16_OVFL);
  }
  for (int lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = (d[0][lane] & ~(0xffffU << d_shift)) | (uint32_t(result[lane]) << d_shift);
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_fma_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c, uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  return run(flags, exec_mask, mode, d, a, b, c);
}

int goc_rdna4_v_fmac_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b,
                         uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return goc::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return goc_rdna4_v_fmac_f16(flags, exec_mask, uint32_t(mode), d, source, b, nullptr);
        });
  const uint32_t known = GOC_ALU_NEG_A | GOC_ALU_NEG_B | GOC_ALU_ABS_A | GOC_ALU_ABS_B |
                         GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP | GOC_ALU_HIGH_A | GOC_ALU_HIGH_B |
                         GOC_ALU_HIGH_D;
  if (int error = goc::validate(flags, mode & ~known, true))
    return error;
  // FMAC reads the same destination half that it overwrites.
  if (mode & GOC_ALU_HIGH_D)
    mode |= GOC_ALU_HIGH_C;
  return run(flags, exec_mask, mode, d, a, b, d);
}

int goc_rdna4_v_fmamk_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                          const uint32_t *const *a, uint16_t literal, const uint32_t *const *b,
                          uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<goc::FmaOperands::MultiplyLiteral>(flags, exec_mask, mode, d, a, b, nullptr, literal);
}

int goc_rdna4_v_fmaak_f16(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b, uint16_t literal,
                          uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<goc::FmaOperands::AddLiteral>(flags, exec_mask, mode, d, a, b, nullptr, literal);
}

// SPDX-License-Identifier: MIT

#include "rdna4_boolean.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_dpp.h"

#include <stdint.h>

namespace {

template <goc::Boolean Op, bool Half>
int boolean(uint64_t flags, uint64_t mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b) {
  if (mode >> 32) {
    if constexpr (!Half) {
      return goc::execute_dpp(
          flags, mask, mode, a, [&](uint32_t effective, const uint32_t *const *source) {
            return boolean<Op, Half>(flags, effective, uint32_t(mode), d, source, b);
          });
    } else {
      return GOC_ERROR_INVALID_FLAGS;
    }
  }

  const uint32_t known =
      Half ? GOC_ALU_HIGH_A | GOC_ALU_HIGH_D | (Op == goc::Boolean::Not ? 0 : GOC_ALU_HIGH_B) : 0;
  if (int error = goc::validate(flags, mode & ~known))
    return error;
  if (!uint32_t(mask))
    return GOC_SUCCESS;
  const uint32_t *bp = nullptr;
  if constexpr (Op != goc::Boolean::Not)
    bp = b[0];
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::boolean_x86_64_v4<Op, Half>(uint32_t(mask), mode, d[0], a[0], bp);
    return GOC_SUCCESS;
  }
#endif
  // AVX2 candidates were slower than baseline for these Boolean operations.

  unsigned sa = Half && (mode & GOC_ALU_HIGH_A) ? 16 : 0;
  unsigned sb = Half && (mode & GOC_ALU_HIGH_B) ? 16 : 0;
  unsigned sd = Half && (mode & GOC_ALU_HIGH_D) ? 16 : 0;
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    uint32_t x = a[0][lane] >> sa, y = bp ? bp[lane] >> sb : 0;
    if constexpr (Op == goc::Boolean::And)
      result[lane] = x & y;
    if constexpr (Op == goc::Boolean::Or)
      result[lane] = x | y;
    if constexpr (Op == goc::Boolean::Xor)
      result[lane] = x ^ y;
    if constexpr (Op == goc::Boolean::Xnor)
      result[lane] = ~(x ^ y);
    if constexpr (Op == goc::Boolean::Not)
      result[lane] = ~x;
    if constexpr (Half)
      result[lane] =
          (d[0][lane] & ~(uint32_t(65535) << sd)) | (uint32_t(uint16_t(result[lane])) << sd);
  }
  for (int lane = 0; lane < 32; ++lane)
    if ((mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_and_b32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  return boolean<goc::Boolean::And, false>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_or_b32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  return boolean<goc::Boolean::Or, false>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_xor_b32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  return boolean<goc::Boolean::Xor, false>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_not_b32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a) {
  return boolean<goc::Boolean::Not, false>(flags, exec_mask, instruction_flags, d, a, nullptr);
}

int goc_rdna4_v_and_b16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return boolean<goc::Boolean::And, true>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_or_b16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return boolean<goc::Boolean::Or, true>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_xor_b16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return boolean<goc::Boolean::Xor, true>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_not_b16(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return boolean<goc::Boolean::Not, true>(flags, exec_mask, instruction_flags, d, a, nullptr);
}

int goc_rdna4_v_xnor_b32(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  return boolean<goc::Boolean::Xnor, false>(flags, exec_mask, instruction_flags, d, a, b);
}

// SPDX-License-Identifier: MIT

#include "div_scale.h"
#include "division.h"
#include "goc/goc.h"
#include "internal.h"

#include <cstring>
#include <stdint.h>

namespace {

template <unsigned Width, bool Rdna3 = false>
int run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d, uint32_t *condition,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  const uint32_t known =
      GOC_ALU_NEG_A | GOC_ALU_NEG_B | GOC_ALU_NEG_C | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP;
  if (int error = goc::validate(flags, mode & ~known, true,
                                GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS))
    return error;
  if (!exec_mask) {
    *condition = 0;
    return GOC_SUCCESS;
  }
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    *condition = goc::div_scale_x86_64_v4<Width>(exec_mask, mode, d, a, b, c, !Rdna3, flags);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    *condition = goc::div_scale_x86_64_v3<Width>(exec_mask, mode, d, a, b, c, !Rdna3, flags);
    return GOC_SUCCESS;
  }
#endif
  using T = typename goc::DivisionFormat<Width>::Bits;
  T result[32];
  uint32_t output_mask = 0;
  for (unsigned lane = 0; lane < 32; ++lane) {
    T av = a[0][lane], bv = b[0][lane], cv = c[0][lane];
    if constexpr (Width == 64) {
      av |= T(a[1][lane]) << 32;
      bv |= T(b[1][lane]) << 32;
      cv |= T(c[1][lane]) << 32;
    }
    auto scaled = goc::division_scale<Width>(av, bv, cv, mode, !Rdna3, flags);
    result[lane] = scaled.value;
    output_mask |= uint32_t(scaled.post_scale) << lane;
  }
  for (unsigned reg = 0; reg < (Width == 64 ? 2u : 1u); ++reg)
    for (unsigned lane = 0; lane < 32; ++lane)
      if ((exec_mask >> lane) & 1)
        d[reg][lane] = uint32_t(result[lane] >> (Width == 64 ? 32 * reg : 0));
  *condition = output_mask & exec_mask;
  return GOC_SUCCESS;
}

template <unsigned Width, bool Rdna3 = false>
int run_wave64(uint64_t flags, uint64_t exec_mask, uint32_t mode, uint32_t *const *d,
               uint64_t *condition, const uint32_t *const *a, const uint32_t *const *b,
               const uint32_t *const *c) {
  uint32_t half_condition = 0;
  if (int error =
          run<Width, Rdna3>(flags, 0, mode, nullptr, &half_condition, nullptr, nullptr, nullptr))
    return error;
  uint64_t result_condition = 0;
  if (exec_mask) {
    for (unsigned half = 0; half < 2; ++half) {
      uint32_t *dest[Width / 32];
      const uint32_t *av[Width / 32], *bv[Width / 32], *cv[Width / 32];
      for (unsigned reg = 0; reg < Width / 32; ++reg) {
        dest[reg] = d[reg] + half * 32;
        av[reg] = a[reg] + half * 32;
        bv[reg] = b[reg] + half * 32;
        cv[reg] = c[reg] + half * 32;
      }
      if (int error = run<Width, Rdna3>(flags, uint32_t(exec_mask >> (half * 32)), mode, dest,
                                        &half_condition, av, bv, cv))
        return error;
      result_condition |= uint64_t(half_condition) << (half * 32);
    }
  }
  // Commit after all operand reads and VGPR writes, including aliased storage.
  std::memcpy(condition, &result_condition, sizeof(result_condition));
  return GOC_SUCCESS;
}

} // namespace

int goc_v_div_scale_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, uint32_t *condition, const uint32_t *const *a,
                        const uint32_t *const *b, const uint32_t *const *c) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<32>(flags, exec_mask, instruction_flags, d, condition, a, b, c);
}

int goc_v_div_scale_f64_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, uint32_t *condition, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64>(flags, exec_mask, instruction_flags, d, condition, a, b, c);
}

int goc_v_div_scale_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, uint64_t *condition, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run_wave64<32>(flags, exec_mask, instruction_flags, d, condition, a, b, c);
}

int goc_v_div_scale_f64_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, uint64_t *condition,
                                     const uint32_t *const *a, const uint32_t *const *b,
                                     const uint32_t *const *c) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run_wave64<64>(flags, exec_mask, instruction_flags, d, condition, a, b, c);
}

int goc_v_div_scale_f64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, uint32_t *condition, const uint32_t *const *a,
                        const uint32_t *const *b, const uint32_t *const *c) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run<64, true>(flags, exec_mask, instruction_flags, d, condition, a, b, c);
}

int goc_v_div_scale_f64_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, uint64_t *condition, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return run_wave64<64, true>(flags, exec_mask, instruction_flags, d, condition, a, b, c);
}

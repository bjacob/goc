// SPDX-License-Identifier: MIT

#include "mullit.h"
#include "alu.h"
#include "dpp.h"
#include "goc/goc.h"
#include "internal.h"

#include <cmath>
#include <stdint.h>

namespace {

template <bool Rdna3>
[[gnu::noinline]] void scalar(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c) {
  uint32_t result[32];
  for (unsigned lane = 0; lane < 32; ++lane) {
    float x = goc::alu_input(a[0][lane], mode);
    float y = goc::alu_input(b[0][lane], mode >> 1);
    float z = goc::alu_input(c[0][lane], mode >> 2);
    uint32_t ybits = goc::as_bits(y);
    float value;
    if (ybits == 0xff7fffff || ybits == 0xff800000 || std::isnan(y) || !(z > 0))
      value = goc::as_float(0xff7fffff);
    else
      value = x == 0 || y == 0 ? 0.0f : x * y;
    unsigned scale = (mode >> 6) & 3;
    if (scale) {
      // RDNA4 flushes before scaling; RDNA3 can normalize a tiny product.
      if constexpr (!Rdna3)
        if (!(goc::as_bits(value) & 0x7f800000))
          value = 0;
      value *= scale == 1 ? 2.0f : scale == 2 ? 4.0f : 0.5f;
      uint32_t bits = goc::as_bits(value);
      if (!(bits & 0x7f800000))
        value = Rdna3 ? 0.0f : goc::as_float(bits & 0x80000000);
    }
    if (mode & GOC_ALU_CLAMP)
      value = !(value > 0) ? 0 : value > 1 ? 1 : value;
    result[lane] = goc::as_bits(value);
  }
  for (unsigned lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = result[lane];
}

template <bool Rdna3>
void compute([[maybe_unused]] uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
             const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::mullit_x86_64_v4<Rdna3>(exec_mask, mode, d[0], a[0], b[0], c[0]);
    return;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::mullit_x86_64_v3<Rdna3>(exec_mask, mode, d[0], a[0], b[0], c[0]);
    return;
  }
#endif
  scalar<Rdna3>(exec_mask, mode, d, a, b, c);
}

// Apply explicit guest flushing around the same arithmetic kernels. Compute all
// results before any destination write so every supported alias remains safe.
template <bool Rdna3>
[[gnu::noinline]] void compute_flushed(uint64_t flags, uint32_t exec_mask, uint32_t mode,
                                       uint32_t *const *d, const uint32_t *const *a,
                                       const uint32_t *const *b, const uint32_t *const *c) {
  uint32_t inputs[3][32], result[32];
  const uint32_t *sources[] = {a[0], b[0], c[0]};
  if (flags & GOC_FP_FLUSH_INPUT_DENORMALS)
    for (unsigned operand = 0; operand < 3; ++operand) {
      for (unsigned lane = 0; lane < 32; ++lane) {
        uint32_t bits = sources[operand][lane];
        inputs[operand][lane] = (bits & 0x7fffffff) < 0x00800000 ? bits & 0x80000000 : bits;
      }
      sources[operand] = inputs[operand];
    }
  uint32_t *dest = result;
  compute<Rdna3>(flags, UINT32_MAX, mode, &dest, &sources[0], &sources[1], &sources[2]);
  for (unsigned lane = 0; lane < 32; ++lane) {
    uint32_t bits = result[lane];
    if ((flags & GOC_FP_FLUSH_OUTPUT_DENORMALS) && (bits & 0x7fffffff) < 0x00800000)
      bits &= 0x80000000;
    if ((exec_mask >> lane) & 1)
      d[0][lane] = bits;
  }
}

template <bool Rdna3>
int run(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
        uint32_t *excp_flag_user) {
  if (excp_flag_user && (flags & GOC_SEMANTICS_MASK) != GOC_SEMANTICS_LOOSE)
    return GOC_ERROR_UNSUPPORTED_GLOBAL_STATE;
  if (instruction_flags >> 32)
    return goc::execute_dpp(flags, exec_mask, instruction_flags, a,
                            [&](uint32_t exec_mask, const uint32_t *const *source) {
                              return run<Rdna3>(flags, exec_mask, uint32_t(instruction_flags), d,
                                                source, b, c, nullptr);
                            });
  if (int error = goc::validate(flags, instruction_flags & ~0x1ffU, false,
                                GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS |
                                    GOC_FP_IEEE_MODE))
    return error;
  if constexpr (Rdna3)
    if (!(flags & GOC_FP_FLUSH_OUTPUT_DENORMALS) || (flags & GOC_FP_IEEE_MODE))
      instruction_flags &= ~uint64_t(GOC_ALU_OMOD_HALF);
  if (!exec_mask)
    return GOC_SUCCESS;
  if (flags & (GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS))
    compute_flushed<Rdna3>(flags, exec_mask, instruction_flags, d, a, b, c);
  else
    compute<Rdna3>(flags, exec_mask, instruction_flags, d, a, b, c);
  return GOC_SUCCESS;
}

template <bool Rdna3>
int run_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
               uint32_t *excp_flag_user) {
  if (int error = run<Rdna3>(flags, 0, mode, nullptr, nullptr, nullptr, nullptr, excp_flag_user))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
  for (unsigned half = 0; half < 2; ++half) {
    uint64_t half_mode = mode;
    if (mode & GOC_DPP16) {
      uint64_t rows = (mode & GOC_DPP_ROW_MASK) >> GOC_DPP_ROW_SHIFT;
      half_mode = (mode & ~GOC_DPP_ROW_MASK) | (((rows >> (half * 2)) & 3) << GOC_DPP_ROW_SHIFT);
    }
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *av = a[0] + half * 32, *bv = b[0] + half * 32, *cv = c[0] + half * 32;
    if (int error = run<Rdna3>(flags, uint32_t(exec_mask >> (half * 32)), half_mode, &dest, &av,
                               &bv, &cv, excp_flag_user))
      return error;
  }
  return GOC_SUCCESS;
}

} // namespace

int goc_v_mullit_f32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
                     uint32_t *trapsts) {
  return run<true>(flags, exec_mask, mode, d, a, b, c, trapsts);
}

int goc_v_mullit_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c, uint32_t *trapsts) {
  return run_wave64<true>(flags, exec_mask, mode, d, a, b, c, trapsts);
}

int goc_v_mullit_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, uint32_t *excp_flag_user) {
  return run<false>(flags, exec_mask, mode, d, a, b, c, excp_flag_user);
}

int goc_v_mullit_f32_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c,
                                  uint32_t *excp_flag_user) {
  return run_wave64<false>(flags, exec_mask, mode, d, a, b, c, excp_flag_user);
}

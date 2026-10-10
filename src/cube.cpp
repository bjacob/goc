// SPDX-License-Identifier: MIT

#include "cube.h"
#include "dpp.h"
#include "goc/goc.h"
#include "internal.h"

#include <stdint.h>

namespace {

template <goc::Cube Op, bool Rdna3 = false>
int execute(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (mode >> 32)
    return goc::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return execute<Op, Rdna3>(flags, exec_mask, uint32_t(mode), d, source, b, c);
        });
  // Cube comparisons always flush denormals; only IEEE gates RDNA3 OMOD.
  if constexpr (Rdna3)
    if (flags & GOC_FP_IEEE_MODE)
      mode &= ~GOC_ALU_OMOD_HALF;
  if (int error =
          goc::validate(flags, mode & ~0x1ffU, true,
                        GOC_FP_IEEE_MODE | GOC_FP_FLUSH_INPUT_DENORMALS |
                            GOC_FP_FLUSH_OUTPUT_DENORMALS | (Rdna3 ? GOC_FP_DX10_CLAMP_OFF : 0)))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
  const bool quiet_nan = !Rdna3 || (flags & GOC_FP_IEEE_MODE);
  const bool clamp_nan = !(flags & GOC_FP_DX10_CLAMP_OFF);
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::cube_x86_64_v4<Op>(exec_mask, mode, d[0], a[0], b[0], c[0], quiet_nan, clamp_nan);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
    goc::cube_x86_64_v3<Op>(exec_mask, mode, d[0], a[0], b[0], c[0], quiet_nan, clamp_nan);
    return GOC_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (unsigned lane = 0; lane < 32; ++lane)
    result[lane] =
        goc::cube_value<Op>(a[0][lane], b[0][lane], c[0][lane], mode, quiet_nan, clamp_nan);
  for (unsigned lane = 0; lane < 32; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

template <goc::Cube Op, bool Rdna3 = false>
int execute_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (int error = execute<Op, Rdna3>(flags, 0, mode, nullptr, nullptr, nullptr, nullptr))
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
    if (int error = execute<Op, Rdna3>(flags, uint32_t(exec_mask >> (half * 32)), half_mode, &dest,
                                       &av, &bv, &cv))
      return error;
  }
  return GOC_SUCCESS;
}

} // namespace

int goc_v_cubeid_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  return execute<goc::Cube::Id>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_v_cubesc_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  return execute<goc::Cube::Sc>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_v_cubetc_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  return execute<goc::Cube::Tc>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_v_cubema_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  return execute<goc::Cube::Ma>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_v_cubeid_f32_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c) {
  return execute_wave64<goc::Cube::Id>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_v_cubesc_f32_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c) {
  return execute_wave64<goc::Cube::Sc>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_v_cubetc_f32_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c) {
  return execute_wave64<goc::Cube::Tc>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_v_cubema_f32_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c) {
  return execute_wave64<goc::Cube::Ma>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_v_cubeid_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                     const uint32_t *const *c) {
  return execute<goc::Cube::Id, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_v_cubeid_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  return execute_wave64<goc::Cube::Id, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_v_cubesc_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                     const uint32_t *const *c) {
  return execute<goc::Cube::Sc, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_v_cubesc_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  return execute_wave64<goc::Cube::Sc, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_v_cubetc_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                     const uint32_t *const *c) {
  return execute<goc::Cube::Tc, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_v_cubetc_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  return execute_wave64<goc::Cube::Tc, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_v_cubema_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                     uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                     const uint32_t *const *c) {
  return execute<goc::Cube::Ma, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int goc_v_cubema_f32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c) {
  return execute_wave64<goc::Cube::Ma, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

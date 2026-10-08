// SPDX-License-Identifier: MIT

#include "goc/goc.h"

#include <algorithm>
#include <cfenv>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdint.h>

namespace {

using Instruction = decltype(&goc_rdna4_v_fma_f32);

int log(uint64_t flags, uint64_t mask, uint32_t modifiers, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *, const uint32_t *const *) {
  return goc_rdna4_v_log_f32(flags, mask, modifiers, d, a);
}

struct Case {
  Instruction fn;
  bool wave64, exact;
  uint32_t modifiers;
};

const Case cases[] = {
    {goc_rdna4_v_fma_f32, false, false, 0},
    {log, false, false, 0},
    {goc_rdna4_v_dot2_f32_f16, false, true, 0},
    {goc_rdna4_v_dot2_f32_bf16, false, true, 0},
    {goc_rdna4_v_wmma_f32_16x16x16_f16, false, true, 63},
    {goc_rdna4_v_wmma_f32_16x16x16_bf16, false, true, 63},
    {goc_rdna4_v_wmma_f16_16x16x16_f16, false, true, 63},
    {goc_rdna4_v_wmma_bf16_16x16x16_bf16, false, true, 63},
    {goc_rdna4w64_v_wmma_f32_16x16x16_f16, true, true, 63},
    {goc_rdna4w64_v_wmma_f32_16x16x16_bf16, true, true, 63},
    {goc_rdna4w64_v_wmma_f16_16x16x16_f16, true, true, 63},
    {goc_rdna4w64_v_wmma_bf16_16x16x16_bf16, true, true, 63},
    {goc_rdna4_v_wmma_f32_16x16x16_fp8_fp8, false, false, GOC_WMMA_NEG_C | GOC_WMMA_ABS_C},
    {goc_rdna4_v_wmma_f32_16x16x16_fp8_bf8, false, false, GOC_WMMA_NEG_C | GOC_WMMA_ABS_C},
    {goc_rdna4_v_wmma_f32_16x16x16_bf8_fp8, false, false, GOC_WMMA_NEG_C | GOC_WMMA_ABS_C},
    {goc_rdna4_v_wmma_f32_16x16x16_bf8_bf8, false, false, GOC_WMMA_NEG_C | GOC_WMMA_ABS_C},
    {goc_rdna4_v_wmma_i32_16x16x16_iu8, false, true,
     GOC_WMMA_SIGNED_A | GOC_WMMA_SIGNED_B | GOC_WMMA_CLAMP},
    {goc_rdna4_v_wmma_i32_16x16x16_iu4, false, true,
     GOC_WMMA_SIGNED_A | GOC_WMMA_SIGNED_B | GOC_WMMA_CLAMP},
    {goc_rdna4_v_wmma_i32_16x16x32_iu4, false, true,
     GOC_WMMA_SIGNED_A | GOC_WMMA_SIGNED_B | GOC_WMMA_CLAMP},
};

struct SavedEnvironment {
  std::fenv_t saved;

  SavedEnvironment() { std::fegetenv(&saved); }

  ~SavedEnvironment() { std::fesetenv(&saved); }
};

} // namespace

TEST(ExecMask, EmptyMaskStillValidatesFlagsAndPreservesState) {
  SavedEnvironment environment;
  for (const auto &f : cases)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint64_t mask : {UINT64_C(0), UINT64_C(0xffffffff00000000)}) {
        if (f.wave64 && mask)
          continue;
        uint32_t storage[24][64];
        uint32_t *v[24];
        for (int reg = 0; reg < 24; ++reg) {
          // Signaling NaNs expose accidental FP evaluation on inactive lanes.
          std::fill(storage[reg], storage[reg] + 64, UINT32_C(0x7f800001));
          v[reg] = storage[reg];
        }
        const auto call = [&](uint64_t flags, uint32_t modifiers, int expected) {
          std::feclearexcept(FE_ALL_EXCEPT);
          std::feraiseexcept(FE_DIVBYZERO);
          EXPECT_EQ(f.fn(flags, mask, modifiers, v + 16, v, v + 4, v + 8), expected);
          EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
          for (const auto &reg : storage)
            for (uint32_t bits : reg)
              EXPECT_EQ(bits, UINT32_C(0x7f800001));
        };
        for (uint32_t modifiers : {UINT32_C(0), f.modifiers}) {
          call(cpu, modifiers, GOC_SUCCESS);
          call(cpu | GOC_SEMANTICS_EXACT_EMPIRICAL, modifiers, GOC_SUCCESS);
          call(cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, modifiers,
               f.exact ? GOC_SUCCESS : GOC_ERROR_UNSUPPORTED_SEMANTICS);
          call(cpu | (UINT64_C(2) << 16) | GOC_SEMANTICS_STRICT, modifiers,
               GOC_ERROR_UNSUPPORTED_SEMANTICS);
          call(cpu | (UINT64_C(1) << 63), modifiers, GOC_ERROR_INVALID_FLAGS);
        }
        call(cpu, UINT32_C(1) << 31, GOC_ERROR_INVALID_FLAGS);
      }
}

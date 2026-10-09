// SPDX-License-Identifier: MIT

#include "fp_environment.h"
#include "goc/goc.h"
#include "rdna4_test_instruction.h"

#include <algorithm>
#include <cfenv>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdint.h>

namespace {

using Instruction = goc_test::WaveInstruction<decltype(&goc_rdna4_v_mad_u32_u24)>;

template <auto Fn>
int unary(uint64_t flags, uint32_t exec_mask, uint64_t modifiers, uint32_t *const *d,
          const uint32_t *const *a, const uint32_t *const *, const uint32_t *const *) {
  return goc_test::without_exceptions(Fn, flags, exec_mask, modifiers, d, a);
}

template <auto Fn>
int binary(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
           const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *) {
  return goc_test::without_exceptions(Fn, flags, exec_mask, mode, d, a, b);
}

template <bool Half, bool Multiply>
int literal_fma(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *) {
  if constexpr (Half) {
    if constexpr (Multiply)
      return goc_rdna4_v_fmamk_f16(flags, exec_mask, mode, d, a, 0x3800, b, nullptr);
    return goc_rdna4_v_fmaak_f16(flags, exec_mask, mode, d, a, b, 0x3800, nullptr);
  } else {
    if constexpr (Multiply)
      return goc_rdna4_v_fmamk_f32(flags, exec_mask, mode, d, a, 0x3f000000, b, nullptr);
    return goc_rdna4_v_fmaak_f32(flags, exec_mask, mode, d, a, b, 0x3f000000, nullptr);
  }
}

struct Case {
  Instruction fn;
  bool exact;
  uint32_t modifiers;
};

const Case cases[] = {
    {binary<goc_rdna4_v_pk_add_f16>, false,
     GOC_PK_NEG_LO_B | GOC_PK_NEG_HI_A | GOC_PK_LO_A_HIGH | GOC_PK_HI_B_LOW | GOC_PK_CLAMP},
    {binary<goc_rdna4_v_pk_mul_f16>, false,
     GOC_PK_NEG_LO_B | GOC_PK_NEG_HI_A | GOC_PK_LO_A_HIGH | GOC_PK_HI_B_LOW | GOC_PK_CLAMP},
    {binary<goc_rdna4_v_pk_min_num_f16>, false,
     GOC_PK_NEG_LO_B | GOC_PK_NEG_HI_A | GOC_PK_LO_A_HIGH | GOC_PK_HI_B_LOW | GOC_PK_CLAMP},
    {binary<goc_rdna4_v_pk_max_num_f16>, false,
     GOC_PK_NEG_LO_B | GOC_PK_NEG_HI_A | GOC_PK_LO_A_HIGH | GOC_PK_HI_B_LOW | GOC_PK_CLAMP},
    {binary<goc_rdna4_v_pk_minimum_f16>, false,
     GOC_PK_NEG_LO_B | GOC_PK_NEG_HI_A | GOC_PK_LO_A_HIGH | GOC_PK_HI_B_LOW | GOC_PK_CLAMP},
    {binary<goc_rdna4_v_pk_maximum_f16>, false,
     GOC_PK_NEG_LO_B | GOC_PK_NEG_HI_A | GOC_PK_LO_A_HIGH | GOC_PK_HI_B_LOW | GOC_PK_CLAMP},

    {goc_rdna4_v_pk_fma_f16, true, 0x1fff},
    {binary<goc_rdna4_v_pk_fmac_f16>, true, 0},
    {literal_fma<true, true>, true, GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_D},
    {literal_fma<true, false>, true, GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_D},
    {literal_fma<false, true>, false, 0},
    {literal_fma<false, false>, false, 0},

    {goc_rdna4_v_min3_num_f16, false, 0x1fffU},
    {goc_rdna4_v_max3_num_f16, false, 0x1fffU},
    {goc_rdna4_v_minmax_num_f16, false, 0x1fffU},
    {goc_rdna4_v_maxmin_num_f16, false, 0x1fffU},
    {goc_rdna4_v_minimum3_f16, false, 0x1fffU},
    {goc_rdna4_v_maximum3_f16, false, 0x1fffU},
    {goc_rdna4_v_minimummaximum_f16, false, 0x1fffU},
    {goc_rdna4_v_maximumminimum_f16, false, 0x1fffU},
    {goc_rdna4_v_med3_num_f16, false, 0x1fffU},

    {binary<goc_rdna4_v_ldexp_f16>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP | GOC_ALU_HIGH_A |
         GOC_ALU_HIGH_B | GOC_ALU_HIGH_D},
    {unary<goc_rdna4_v_frexp_exp_i16_f16>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP | GOC_ALU_HIGH_A |
         GOC_ALU_HIGH_D},
    {unary<goc_rdna4_v_trunc_f16>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP | GOC_ALU_HIGH_A |
         GOC_ALU_HIGH_D},
    {unary<goc_rdna4_v_ceil_f16>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP | GOC_ALU_HIGH_A |
         GOC_ALU_HIGH_D},
    {unary<goc_rdna4_v_rndne_f16>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP | GOC_ALU_HIGH_A |
         GOC_ALU_HIGH_D},
    {unary<goc_rdna4_v_floor_f16>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP | GOC_ALU_HIGH_A |
         GOC_ALU_HIGH_D},
    {unary<goc_rdna4_v_sqrt_f16>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP | GOC_ALU_HIGH_A |
         GOC_ALU_HIGH_D},
    {unary<goc_rdna4_v_rcp_f16>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP | GOC_ALU_HIGH_A |
         GOC_ALU_HIGH_D},
    {unary<goc_rdna4_v_rsq_f16>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP | GOC_ALU_HIGH_A |
         GOC_ALU_HIGH_D},
    {unary<goc_rdna4_v_exp_f16>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP | GOC_ALU_HIGH_A |
         GOC_ALU_HIGH_D},
    {unary<goc_rdna4_v_log_f16>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP | GOC_ALU_HIGH_A |
         GOC_ALU_HIGH_D},
    {unary<goc_rdna4_v_fract_f16>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP | GOC_ALU_HIGH_A |
         GOC_ALU_HIGH_D},
    {unary<goc_rdna4_v_frexp_mant_f16>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP | GOC_ALU_HIGH_A |
         GOC_ALU_HIGH_D},

    {binary<goc_rdna4_v_add_f16>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_B | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP | GOC_ALU_HIGH_A |
         GOC_ALU_HIGH_B | GOC_ALU_HIGH_D},
    {binary<goc_rdna4_v_sub_f16>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_B | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP | GOC_ALU_HIGH_A |
         GOC_ALU_HIGH_B | GOC_ALU_HIGH_D},
    {binary<goc_rdna4_v_subrev_f16>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_B | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP | GOC_ALU_HIGH_A |
         GOC_ALU_HIGH_B | GOC_ALU_HIGH_D},
    {binary<goc_rdna4_v_mul_f16>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_B | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP | GOC_ALU_HIGH_A |
         GOC_ALU_HIGH_B | GOC_ALU_HIGH_D},
    {binary<goc_rdna4_v_min_num_f16>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_B | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP | GOC_ALU_HIGH_A |
         GOC_ALU_HIGH_B | GOC_ALU_HIGH_D},
    {binary<goc_rdna4_v_max_num_f16>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_B | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP | GOC_ALU_HIGH_A |
         GOC_ALU_HIGH_B | GOC_ALU_HIGH_D},
    {binary<goc_rdna4_v_minimum_f16>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_B | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP | GOC_ALU_HIGH_A |
         GOC_ALU_HIGH_B | GOC_ALU_HIGH_D},
    {binary<goc_rdna4_v_maximum_f16>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_B | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP | GOC_ALU_HIGH_A |
         GOC_ALU_HIGH_B | GOC_ALU_HIGH_D},

    {binary<goc_rdna4_v_add_nc_u32>, false, GOC_ALU_CLAMP},
    {binary<goc_rdna4_v_sub_nc_u32>, false, GOC_ALU_CLAMP},
    {binary<goc_rdna4_v_subrev_nc_u32>, false, GOC_ALU_CLAMP},
    {binary<goc_rdna4_v_add_nc_i32>, false, GOC_ALU_CLAMP},
    {binary<goc_rdna4_v_sub_nc_i32>, false, GOC_ALU_CLAMP},
    {goc_rdna4_v_add3_u32, false, 0},

    {binary<goc_rdna4_v_mul_lo_u32>, false, 0},
    {binary<goc_rdna4_v_mul_hi_u32>, false, 0},
    {binary<goc_rdna4_v_mul_hi_i32>, false, 0},
    {binary<goc_rdna4_v_mul_i32_i24>, false, GOC_ALU_CLAMP},
    {binary<goc_rdna4_v_mul_hi_i32_i24>, false, 0},
    {binary<goc_rdna4_v_mul_u32_u24>, false, GOC_ALU_CLAMP},
    {binary<goc_rdna4_v_mul_hi_u32_u24>, false, 0},

    {binary<goc_rdna4_v_min_i32>, false, 0},
    {binary<goc_rdna4_v_max_i32>, false, 0},
    {goc_rdna4_v_min3_i32, false, 0},
    {goc_rdna4_v_max3_i32, false, 0},
    {goc_rdna4_v_minmax_i32, false, 0},
    {goc_rdna4_v_maxmin_i32, false, 0},
    {goc_rdna4_v_med3_i32, false, 0},
    {binary<goc_rdna4_v_min_u32>, false, 0},
    {binary<goc_rdna4_v_max_u32>, false, 0},
    {goc_rdna4_v_min3_u32, false, 0},
    {goc_rdna4_v_max3_u32, false, 0},
    {goc_rdna4_v_minmax_u32, false, 0},
    {goc_rdna4_v_maxmin_u32, false, 0},
    {goc_rdna4_v_med3_u32, false, 0},

    {binary<goc_rdna4_v_ldexp_f32>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},
    {binary<goc_rdna4_v_ldexp_f64>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},

    {unary<goc_rdna4_v_frexp_exp_i32_f32>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},
    {unary<goc_rdna4_v_frexp_exp_i32_f64>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},

    {unary<goc_rdna4_v_frexp_mant_f32>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},
    {unary<goc_rdna4_v_frexp_mant_f64>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},

    {binary<goc_rdna4_v_min_num_f64>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_B | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},
    {binary<goc_rdna4_v_max_num_f64>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_B | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},
    {binary<goc_rdna4_v_minimum_f64>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_B | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},
    {binary<goc_rdna4_v_maximum_f64>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_B | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},

    {unary<goc_rdna4_v_trunc_f64>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},
    {unary<goc_rdna4_v_ceil_f64>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},
    {unary<goc_rdna4_v_rndne_f64>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},
    {unary<goc_rdna4_v_floor_f64>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},
    {unary<goc_rdna4_v_fract_f64>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},
    {unary<goc_rdna4_v_sqrt_f64>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},
    {unary<goc_rdna4_v_rcp_f64>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},
    {unary<goc_rdna4_v_rsq_f64>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},

    {binary<goc_rdna4_v_add_f64>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_B | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},
    {binary<goc_rdna4_v_mul_f64>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_B | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},
    {goc_rdna4_v_fma_f64, false, 0x1ff},
    {goc_rdna4_v_med3_num_f32, false, 0x1ff},
    {binary<goc_rdna4_v_mul_dx9_zero_f32>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_B | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},
    {unary<goc_rdna4_v_fract_f32>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},
    {goc_rdna4_v_min3_num_f32, false, 0x1ff},
    {goc_rdna4_v_max3_num_f32, false, 0x1ff},
    {goc_rdna4_v_minmax_num_f32, false, 0x1ff},
    {goc_rdna4_v_maxmin_num_f32, false, 0x1ff},
    {goc_rdna4_v_minimum3_f32, false, 0x1ff},
    {goc_rdna4_v_maximum3_f32, false, 0x1ff},
    {goc_rdna4_v_minimummaximum_f32, false, 0x1ff},
    {goc_rdna4_v_maximumminimum_f32, false, 0x1ff},

    {binary<goc_rdna4_v_min_num_f32>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_B | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},
    {binary<goc_rdna4_v_max_num_f32>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_B | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},
    {binary<goc_rdna4_v_minimum_f32>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_B | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},
    {binary<goc_rdna4_v_maximum_f32>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_B | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},

    {binary<goc_rdna4_v_add_f32>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_B | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},
    {binary<goc_rdna4_v_sub_f32>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_B | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},
    {binary<goc_rdna4_v_subrev_f32>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_B | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},
    {binary<goc_rdna4_v_mul_f32>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_B | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},

    {goc_rdna4_v_dot4_i32_iu8, true, GOC_DOT_SIGNED_A | GOC_DOT_SIGNED_B | GOC_DOT_CLAMP},
    {goc_rdna4_v_dot8_i32_iu4, true, GOC_DOT_SIGNED_A | GOC_DOT_SIGNED_B | GOC_DOT_CLAMP},
    {goc_rdna4_v_dot4_u32_u8, true, GOC_DOT_CLAMP},
    {goc_rdna4_v_dot8_u32_u4, true, GOC_DOT_CLAMP},
    {goc_rdna4_v_dot4_f32_fp8_fp8, false, GOC_DOT_NEG_C | GOC_DOT_ABS_C},
    {goc_rdna4_v_dot4_f32_fp8_bf8, false, GOC_DOT_NEG_C | GOC_DOT_ABS_C},
    {goc_rdna4_v_dot4_f32_bf8_fp8, false, GOC_DOT_NEG_C | GOC_DOT_ABS_C},
    {goc_rdna4_v_dot4_f32_bf8_bf8, false, GOC_DOT_NEG_C | GOC_DOT_ABS_C},
    {goc_rdna4_v_dot2_f16_f16, false, 63 | GOC_ALU_HIGH_C | GOC_ALU_HIGH_D},
    {goc_rdna4_v_dot2_bf16_bf16, false, 63 | GOC_ALU_HIGH_C | GOC_ALU_HIGH_D},
    {binary<goc_rdna4_v_fmac_f16>, true,
     GOC_ALU_ABS_A | GOC_ALU_NEG_B | GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_D |
         GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},
    {binary<goc_rdna4_v_fmac_f32>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_B | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},
    {goc_rdna4_v_fma_f16, true, 0x1fff},
    {goc_rdna4_v_fma_f32, false, 0x1ff},
    {goc_rdna4_v_fma_dx9_zero_f32, false, 0x1ff},
    {unary<goc_rdna4_v_log_f32>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},
    {unary<goc_rdna4_v_exp_f32>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},
    {unary<goc_rdna4_v_sqrt_f32>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},
    {unary<goc_rdna4_v_rcp_f32>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},
    {unary<goc_rdna4_v_rsq_f32>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},
    {unary<goc_rdna4_v_floor_f32>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},
    {unary<goc_rdna4_v_ceil_f32>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},
    {unary<goc_rdna4_v_trunc_f32>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},
    {unary<goc_rdna4_v_rndne_f32>, false,
     GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP},
    {goc_rdna4_v_dot2_f32_f16, true, 0x7df},
    {goc_rdna4_v_dot2_f32_bf16, true, 0x7df},
    {goc_rdna4_v_wmma_f32_16x16x16_f16, true, 63},
    {goc_rdna4_v_wmma_f32_16x16x16_bf16, true, 63},
    {goc_rdna4_v_wmma_f16_16x16x16_f16, true, 63},
    {goc_rdna4_v_wmma_bf16_16x16x16_bf16, true, 63},
    {goc_rdna4w64_v_wmma_f32_16x16x16_f16, true, 63},
    {goc_rdna4w64_v_wmma_f32_16x16x16_bf16, true, 63},
    {goc_rdna4w64_v_wmma_f16_16x16x16_f16, true, 63},
    {goc_rdna4w64_v_wmma_bf16_16x16x16_bf16, true, 63},
    {goc_rdna4_v_wmma_f32_16x16x16_fp8_fp8, false, GOC_WMMA_NEG_C | GOC_WMMA_ABS_C},
    {goc_rdna4_v_wmma_f32_16x16x16_fp8_bf8, false, GOC_WMMA_NEG_C | GOC_WMMA_ABS_C},
    {goc_rdna4_v_wmma_f32_16x16x16_bf8_fp8, false, GOC_WMMA_NEG_C | GOC_WMMA_ABS_C},
    {goc_rdna4_v_wmma_f32_16x16x16_bf8_bf8, false, GOC_WMMA_NEG_C | GOC_WMMA_ABS_C},
    {goc_rdna4_v_wmma_i32_16x16x16_iu8, true,
     GOC_WMMA_SIGNED_A | GOC_WMMA_SIGNED_B | GOC_WMMA_CLAMP},
    {goc_rdna4_v_wmma_i32_16x16x16_iu4, true,
     GOC_WMMA_SIGNED_A | GOC_WMMA_SIGNED_B | GOC_WMMA_CLAMP},
    {goc_rdna4_v_wmma_i32_16x16x32_iu4, true,
     GOC_WMMA_SIGNED_A | GOC_WMMA_SIGNED_B | GOC_WMMA_CLAMP},
};

} // namespace

TEST(ExecMask, EmptyMaskStillValidatesFlagsAndPreservesState) {
  goc_test::ScopedFpEnvironment environment;
  ASSERT_TRUE(environment.saved());
  for (const auto &f : cases)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      const uint32_t exec_mask = 0;
      uint32_t storage[24][64];
      uint32_t *v[24];
      for (int reg = 0; reg < 24; ++reg) {
        // Signaling NaNs expose accidental FP evaluation on inactive lanes.
        std::fill(storage[reg], storage[reg] + 64, 0x7f800001U);
        v[reg] = storage[reg];
      }
      const auto call = [&](uint64_t flags, uint32_t modifiers, int expected) {
        std::feclearexcept(FE_ALL_EXCEPT);
        std::feraiseexcept(FE_DIVBYZERO);
        EXPECT_EQ(f.fn(flags, exec_mask, modifiers, v + 16, v, v + 4, v + 8), expected);
        EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
        for (const auto &reg : storage)
          for (uint32_t bits : reg)
            EXPECT_EQ(bits, 0x7f800001U);
      };
      for (uint32_t modifiers : {0U, f.modifiers}) {
        call(cpu, modifiers, GOC_SUCCESS);
        call(cpu | GOC_SEMANTICS_EXACT_EMPIRICAL, modifiers, GOC_SUCCESS);
        call(cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, modifiers,
             f.exact ? GOC_SUCCESS : GOC_ERROR_UNSUPPORTED_SEMANTICS);
        call(cpu | (2ULL << 16) | GOC_SEMANTICS_STRICT, modifiers, GOC_ERROR_UNSUPPORTED_SEMANTICS);
        call(cpu | (1ULL << 63), modifiers, GOC_ERROR_INVALID_FLAGS);
      }
      call(cpu, 1U << 31, GOC_ERROR_INVALID_FLAGS);
    }
}

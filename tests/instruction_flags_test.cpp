// SPDX-License-Identifier: MIT

#include "bitfield_reference.h"
#include "boolean_reference.h"
#include "conversion16_reference.h"
#include "conversion32_reference.h"
#include "dpp_arithmetic_reference.h"
#include "dpp_integer_reference.h"
#include "float_compare_reference.h"
#include "goc/goc.h"
#include "half_binary_reference.h"
#include "half_minmax_reference.h"
#include "half_unary_reference.h"
#include "integer16_reference.h"
#include "integer16_ternary_reference.h"
#include "integer_add_reference.h"
#include "integer_compare_reference.h"
#include "integer_mad_reference.h"
#include "integer_minmax_reference.h"
#include "integer_mul_reference.h"
#include "integer_ternary_reference.h"
#include "unary_reference.h"

#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>
#include <string>
#include <type_traits>

namespace {

// State-only scalar instructions have an immediate before the optional output.
void check_high_flags(const char *name,
                      int (*instruction)(uint64_t, uint64_t, uint16_t, uint32_t *)) {
  SCOPED_TRACE(name);
  for (unsigned bit = 32; bit < 64; ++bit)
    EXPECT_EQ(instruction(0, 1ULL << bit, 0, nullptr), GOC_ERROR_INVALID_FLAGS) << bit;
}

// EXEC-independent instructions place instruction_flags immediately after flags.
template <typename Output, typename... Operands>
void check_high_flags(const char *name,
                      int (*instruction)(uint64_t, uint64_t, Output *, Operands...)) {
  SCOPED_TRACE(name);
  for (unsigned bit = 32; bit < 64; ++bit)
    EXPECT_EQ(instruction(0, 1ULL << bit, static_cast<Output *>(nullptr), Operands{}...),
              GOC_ERROR_INVALID_FLAGS)
        << bit;
}

template <typename Mask, typename... Operands>
void check_high_flags(const char *name, int (*instruction)(uint64_t, Mask, uint64_t, Operands...)) {
  SCOPED_TRACE(name);
  std::string base_name(name);
  if (base_name.size() >= 7 && base_name.compare(base_name.size() - 7, 7, "_wave64") == 0) {
    EXPECT_TRUE((std::is_same_v<Mask, uint64_t>));
    base_name.resize(base_name.size() - 7);
    // Only these audited families currently share DPP support across widths.
    if (base_name == "goc_v_dot2_f32_f16_rdna4" || base_name == "goc_v_dot2_f32_bf16_rdna4" ||
        base_name == "goc_v_dot2_f32_f16" || base_name == "goc_v_dot2_f32_bf16" ||
        base_name == "goc_v_dot2acc_f32_f16" || base_name == "goc_v_dot2_f16_f16" ||
        base_name == "goc_v_dot2_bf16_bf16" || base_name == "goc_v_mullit_f32" ||
        base_name == "goc_v_cubeid_f32" || base_name == "goc_v_cubesc_f32" ||
        base_name == "goc_v_cubetc_f32" || base_name == "goc_v_cubema_f32" ||
        base_name == "goc_v_rcp_iflag_f32" || base_name == "goc_v_div_fixup_f16" ||
        base_name == "goc_v_sin_f32" || base_name == "goc_v_cos_f32" ||
        base_name == "goc_v_sin_f16" || base_name == "goc_v_cos_f16" ||
        base_name == "goc_v_trunc_f32" || base_name == "goc_v_ceil_f32" ||
        base_name == "goc_v_rndne_f32" || base_name == "goc_v_floor_f32" ||
        base_name == "goc_v_sqrt_f32" || base_name == "goc_v_rcp_f32" ||
        base_name == "goc_v_rsq_f32" || base_name == "goc_v_exp_f32" ||
        base_name == "goc_v_log_f32" || base_name == "goc_v_fract_f32" ||
        base_name == "goc_v_frexp_mant_f32" || base_name == "goc_v_trunc_f16" ||
        base_name == "goc_v_ceil_f16" || base_name == "goc_v_rndne_f16" ||
        base_name == "goc_v_floor_f16" || base_name == "goc_v_sqrt_f16" ||
        base_name == "goc_v_rcp_f16" || base_name == "goc_v_rsq_f16" ||
        base_name == "goc_v_exp_f16" || base_name == "goc_v_log_f16" ||
        base_name == "goc_v_fract_f16" || base_name == "goc_v_frexp_mant_f16" ||
        base_name == "goc_v_min3_num_f32" || base_name == "goc_v_max3_num_f32" ||
        base_name == "goc_v_minmax_num_f32" || base_name == "goc_v_maxmin_num_f32" ||
        base_name == "goc_v_minimum3_f32" || base_name == "goc_v_maximum3_f32" ||
        base_name == "goc_v_minimummaximum_f32" || base_name == "goc_v_maximumminimum_f32" ||
        base_name == "goc_v_med3_num_f32" || base_name == "goc_v_min3_f32" ||
        base_name == "goc_v_max3_f32" || base_name == "goc_v_minmax_f32" ||
        base_name == "goc_v_maxmin_f32" || base_name == "goc_v_med3_f32" ||
        base_name == "goc_v_min3_num_f16" || base_name == "goc_v_max3_num_f16" ||
        base_name == "goc_v_minmax_num_f16" || base_name == "goc_v_maxmin_num_f16" ||
        base_name == "goc_v_minimum3_f16" || base_name == "goc_v_maximum3_f16" ||
        base_name == "goc_v_minimummaximum_f16" || base_name == "goc_v_maximumminimum_f16" ||
        base_name == "goc_v_med3_num_f16" || base_name == "goc_v_min3_f16" ||
        base_name == "goc_v_max3_f16" || base_name == "goc_v_minmax_f16" ||
        base_name == "goc_v_maxmin_f16" || base_name == "goc_v_med3_f16" ||
        base_name == "goc_v_add_f16" || base_name == "goc_v_add_f16_rdna4" ||
        base_name == "goc_v_sub_f16" || base_name == "goc_v_sub_f16_rdna4" ||
        base_name == "goc_v_subrev_f16" || base_name == "goc_v_subrev_f16_rdna4" ||
        base_name == "goc_v_mul_f16" || base_name == "goc_v_mul_f16_rdna4" ||
        base_name == "goc_v_min_num_f16" || base_name == "goc_v_max_num_f16" ||
        base_name == "goc_v_minimum_f16" || base_name == "goc_v_maximum_f16" ||
        base_name == "goc_v_min_f16" || base_name == "goc_v_max_f16" ||
        base_name == "goc_v_fma_mix_f32" || base_name == "goc_v_fma_mixlo_f16" ||
        base_name == "goc_v_fma_mixhi_f16" || base_name == "goc_v_fma_mix_f32_rdna4" ||
        base_name == "goc_v_fma_mixlo_f16_rdna4" || base_name == "goc_v_fma_mixhi_f16_rdna4" ||
        base_name == "goc_v_fma_f16" || base_name == "goc_v_fma_f16_rdna4" ||
        base_name == "goc_v_fmac_f16" || base_name == "goc_v_fmac_f16_rdna4" ||
        base_name == "goc_v_add_f32" || base_name == "goc_v_add_f32_rdna4" ||
        base_name == "goc_v_sub_f32" || base_name == "goc_v_sub_f32_rdna4" ||
        base_name == "goc_v_subrev_f32" || base_name == "goc_v_subrev_f32_rdna4" ||
        base_name == "goc_v_mul_f32" || base_name == "goc_v_mul_f32_rdna4" ||
        base_name == "goc_v_min_num_f32" || base_name == "goc_v_max_num_f32" ||
        base_name == "goc_v_minimum_f32" || base_name == "goc_v_maximum_f32" ||
        base_name == "goc_v_mul_dx9_zero_f32" || base_name == "goc_v_mul_dx9_zero_f32_rdna4" ||
        base_name == "goc_v_min_f32" || base_name == "goc_v_max_f32" ||
        base_name == "goc_v_fma_f32" || base_name == "goc_v_fma_f32_rdna4" ||
        base_name == "goc_v_fmac_f32" || base_name == "goc_v_fmac_f32_rdna4" ||
        base_name == "goc_v_fma_dx9_zero_f32" || base_name == "goc_v_fma_dx9_zero_f32_rdna4" ||
        base_name == "goc_v_fmac_dx9_zero_f32" || base_name == "goc_v_ldexp_f32" ||
        base_name == "goc_v_ldexp_f16" || base_name == "goc_v_frexp_exp_i16_f16" ||
        base_name.compare(0, 9, "goc_v_cmp") == 0 || base_name == "goc_v_mov_b32" ||
        base_name == "goc_v_mov_b16" || base_name == "goc_v_mbcnt_lo_u32_b32" ||
        base_name == "goc_v_mbcnt_hi_u32_b32" || base_name == "goc_v_clz_i32_u32" ||
        base_name == "goc_v_ctz_i32_b32" || base_name == "goc_v_cls_i32" ||
        base_name == "goc_v_bcnt_u32_b32" || base_name == "goc_v_and_b32" ||
        base_name == "goc_v_or_b32" || base_name == "goc_v_xor_b32" ||
        base_name == "goc_v_xnor_b32" || base_name == "goc_v_not_b32" ||
        base_name == "goc_v_and_b16" || base_name == "goc_v_or_b16" ||
        base_name == "goc_v_xor_b16" || base_name == "goc_v_not_b16" ||
        base_name == "goc_v_cndmask_b16" || base_name == "goc_v_cndmask_b32" ||
        base_name == "goc_v_add_co_u32" || base_name == "goc_v_add_co_ci_u32" ||
        base_name == "goc_v_sub_co_u32" || base_name == "goc_v_sub_co_ci_u32" ||
        base_name == "goc_v_subrev_co_u32" || base_name == "goc_v_subrev_co_ci_u32" ||
        base_name == "goc_v_lshlrev_b32" || base_name == "goc_v_lshrrev_b32" ||
        base_name == "goc_v_ashrrev_i32" || base_name == "goc_v_lshlrev_b64" ||
        base_name == "goc_v_lshrrev_b64" || base_name == "goc_v_ashrrev_i64" ||
        base_name == "goc_v_bfe_u32" || base_name == "goc_v_bfe_i32" ||
        base_name == "goc_v_bfi_b32" || base_name == "goc_v_bfm_b32" ||
        base_name == "goc_v_bfrev_b32" || base_name == "goc_v_alignbit_b32" ||
        base_name == "goc_v_alignbyte_b32" || base_name == "goc_v_perm_b32" ||
        base_name == "goc_v_add_nc_u32" || base_name == "goc_v_sub_nc_u32" ||
        base_name == "goc_v_subrev_nc_u32" || base_name == "goc_v_add_nc_i32" ||
        base_name == "goc_v_sub_nc_i32" || base_name == "goc_v_add3_u32" ||
        base_name == "goc_v_mul_lo_u32" || base_name == "goc_v_mul_hi_u32" ||
        base_name == "goc_v_mul_hi_i32" || base_name == "goc_v_mul_i32_i24" ||
        base_name == "goc_v_mul_hi_i32_i24" || base_name == "goc_v_mul_u32_u24" ||
        base_name == "goc_v_mul_hi_u32_u24" || base_name == "goc_v_min_i32" ||
        base_name == "goc_v_max_i32" || base_name == "goc_v_min3_i32" ||
        base_name == "goc_v_max3_i32" || base_name == "goc_v_minmax_i32" ||
        base_name == "goc_v_maxmin_i32" || base_name == "goc_v_med3_i32" ||
        base_name == "goc_v_min_u32" || base_name == "goc_v_max_u32" ||
        base_name == "goc_v_min3_u32" || base_name == "goc_v_max3_u32" ||
        base_name == "goc_v_minmax_u32" || base_name == "goc_v_maxmin_u32" ||
        base_name == "goc_v_med3_u32" || base_name == "goc_v_lshl_add_u32" ||
        base_name == "goc_v_add_lshl_u32" || base_name == "goc_v_lshl_or_b32" ||
        base_name == "goc_v_and_or_b32" || base_name == "goc_v_or3_b32" ||
        base_name == "goc_v_xor3_b32" || base_name == "goc_v_xad_u32" ||
        base_name == "goc_v_lerp_u8" || base_name == "goc_v_pk_add_i16" ||
        base_name == "goc_v_pk_sub_i16" || base_name == "goc_v_pk_add_u16" ||
        base_name == "goc_v_pk_sub_u16" || base_name == "goc_v_pk_min_i16" ||
        base_name == "goc_v_pk_max_i16" || base_name == "goc_v_pk_min_u16" ||
        base_name == "goc_v_pk_max_u16" || base_name == "goc_v_pk_mul_lo_u16" ||
        base_name == "goc_v_pk_lshlrev_b16" || base_name == "goc_v_pk_lshrrev_b16" ||
        base_name == "goc_v_pk_ashrrev_i16" || base_name == "goc_v_add_nc_i16" ||
        base_name == "goc_v_sub_nc_i16" || base_name == "goc_v_add_nc_u16" ||
        base_name == "goc_v_sub_nc_u16" || base_name == "goc_v_min_i16" ||
        base_name == "goc_v_max_i16" || base_name == "goc_v_min_u16" ||
        base_name == "goc_v_max_u16" || base_name == "goc_v_mul_lo_u16" ||
        base_name == "goc_v_lshlrev_b16" || base_name == "goc_v_lshrrev_b16" ||
        base_name == "goc_v_ashrrev_i16" || base_name == "goc_v_pk_mad_i16" ||
        base_name == "goc_v_pk_mad_u16" || base_name == "goc_v_mad_u16" ||
        base_name == "goc_v_mad_i16" || base_name == "goc_v_min3_u16" ||
        base_name == "goc_v_min3_i16" || base_name == "goc_v_max3_u16" ||
        base_name == "goc_v_max3_i16" || base_name == "goc_v_med3_u16" ||
        base_name == "goc_v_med3_i16" || base_name == "goc_v_mad_u32_u16" ||
        base_name == "goc_v_mad_i32_i16" || base_name == "goc_v_mad_u32_u24" ||
        base_name == "goc_v_mad_i32_i24" || base_name == "goc_v_mad_co_u64_u32" ||
        base_name == "goc_v_mad_co_i64_i32" || base_name == "goc_v_mad_u64_u32" ||
        base_name == "goc_v_mad_i64_i32" || base_name == "goc_v_cvt_f16_i16" ||
        base_name == "goc_v_cvt_f16_u16" || base_name == "goc_v_cvt_i16_f16" ||
        base_name == "goc_v_cvt_u16_f16" || base_name == "goc_v_cvt_f16_f32" ||
        base_name == "goc_v_cvt_f32_f16" || base_name == "goc_v_cvt_f32_i32" ||
        base_name == "goc_v_cvt_f32_u32" || base_name == "goc_v_cvt_i32_f32" ||
        base_name == "goc_v_cvt_u32_f32" || base_name == "goc_v_cvt_nearest_i32_f32" ||
        base_name == "goc_v_cvt_floor_i32_f32" || base_name == "goc_v_cvt_pk_rtz_f16_f32" ||
        base_name == "goc_v_cvt_pk_i16_f32" || base_name == "goc_v_cvt_pk_u16_f32" ||
        base_name == "goc_v_cvt_pk_norm_i16_f32" || base_name == "goc_v_cvt_pk_norm_u16_f32" ||
        base_name == "goc_v_cvt_pk_norm_i16_f16" || base_name == "goc_v_cvt_pk_norm_u16_f16" ||
        base_name == "goc_v_cvt_norm_i16_f16" || base_name == "goc_v_cvt_norm_u16_f16" ||
        base_name == "goc_v_cvt_pk_fp8_f32" || base_name == "goc_v_cvt_pk_bf8_f32" ||
        base_name == "goc_v_cvt_sr_fp8_f32" || base_name == "goc_v_cvt_sr_bf8_f32" ||
        base_name == "goc_v_cvt_f32_fp8" || base_name == "goc_v_cvt_f32_bf8" ||
        base_name == "goc_v_cvt_pk_f32_fp8" || base_name == "goc_v_cvt_pk_f32_bf8" ||
        base_name == "goc_v_cvt_f32_ubyte0" || base_name == "goc_v_cvt_f32_ubyte1" ||
        base_name == "goc_v_cvt_f32_ubyte2" || base_name == "goc_v_cvt_f32_ubyte3" ||
        base_name == "goc_v_cvt_off_f32_i4" || base_name == "goc_v_cvt_pk_u8_f32" ||
        base_name == "goc_v_frexp_exp_i32_f32" || base_name == "goc_v_frexp_exp_i32_f64" ||
        base_name == "goc_v_cvt_i32_i16" || base_name == "goc_v_cvt_u32_u16" ||
        base_name == "goc_v_cvt_pk_i16_i32" || base_name == "goc_v_cvt_pk_u16_u32" ||
        base_name == "goc_v_sat_pk_u8_i16" || base_name == "goc_v_pack_b32_f16" ||
        base_name == "goc_v_sad_u8" || base_name == "goc_v_sad_hi_u8" ||
        base_name == "goc_v_sad_u16" || base_name == "goc_v_sad_u32" ||
        base_name == "goc_v_msad_u8" || base_name == "goc_v_qsad_pk_u16_u8" ||
        base_name == "goc_v_mqsad_pk_u16_u8" || base_name == "goc_v_mqsad_u32_u8" ||
        base_name == "goc_v_rcp_iflag_f32_rdna4" || base_name == "goc_v_trunc_f32_rdna4" ||
        base_name == "goc_v_ceil_f32_rdna4" || base_name == "goc_v_rndne_f32_rdna4" ||
        base_name == "goc_v_frexp_mant_f32_rdna4" || base_name == "goc_v_floor_f32_rdna4" ||
        base_name == "goc_v_sqrt_f32_rdna4" || base_name == "goc_v_rcp_f32_rdna4" ||
        base_name == "goc_v_rsq_f32_rdna4" || base_name == "goc_v_exp_f32_rdna4" ||
        base_name == "goc_v_log_f32_rdna4")
      name = base_name.c_str();
  } else {
    EXPECT_TRUE((std::is_same_v<Mask, uint32_t>));
  }
  bool supports_dpp =
      std::strcmp(name, "goc_v_add_f16_rdna4") == 0 ||
      std::strcmp(name, "goc_v_sub_f16_rdna4") == 0 ||
      std::strcmp(name, "goc_v_subrev_f16_rdna4") == 0 ||
      std::strcmp(name, "goc_v_mul_f16_rdna4") == 0 ||
      std::strcmp(name, "goc_v_mul_dx9_zero_f32_rdna4") == 0 ||
      std::strcmp(name, "goc_v_add_f32_rdna4") == 0 ||
      std::strcmp(name, "goc_v_sub_f32_rdna4") == 0 ||
      std::strcmp(name, "goc_v_subrev_f32_rdna4") == 0 ||
      std::strcmp(name, "goc_v_mul_f32_rdna4") == 0 ||
      std::strcmp(name, "goc_v_dot2acc_f32_f16") == 0 ||
      std::strncmp(name, "goc_v_movrel", 12) == 0 || std::strcmp(name, "goc_v_mov_b32") == 0 ||
      std::strcmp(name, "goc_v_mov_b16") == 0 || std::strcmp(name, "goc_v_fma_f32") == 0 ||
      std::strcmp(name, "goc_v_fma_f32_rdna4") == 0 || std::strcmp(name, "goc_v_fmac_f32") == 0 ||
      std::strcmp(name, "goc_v_fmac_f32_rdna4") == 0 || std::strcmp(name, "goc_v_fma_f16") == 0 ||
      std::strcmp(name, "goc_v_fma_f16_rdna4") == 0 || std::strcmp(name, "goc_v_fmac_f16") == 0 ||
      std::strcmp(name, "goc_v_fmac_f16_rdna4") == 0 || std::strcmp(name, "goc_v_ldexp_f16") == 0 ||
      std::strcmp(name, "goc_v_frexp_exp_i16_f16") == 0 ||
      std::strcmp(name, "goc_v_frexp_exp_i32_f32") == 0 ||
      std::strcmp(name, "goc_v_ldexp_f32") == 0;
  for (auto mnemonic : goc_test::dpp_arithmetic_names)
    supports_dpp |= std::strcmp(name + sizeof("goc_") - 1, mnemonic) == 0;
  for (auto mnemonic : goc_test::unary_names)
    supports_dpp |= std::strcmp(name + sizeof("goc_") - 1, mnemonic) == 0;
  for (auto mnemonic : goc_test::dpp_integer_names)
    supports_dpp |= std::strcmp(name + sizeof("goc_") - 1, mnemonic) == 0;
  for (auto mnemonic : goc_test::integer_minmax_names)
    supports_dpp |= std::strcmp(name + sizeof("goc_") - 1, mnemonic) == 0;
  for (auto mnemonic : goc_test::integer_ternary_names)
    supports_dpp |= std::strcmp(name + sizeof("goc_") - 1, mnemonic) == 0;
  for (auto mnemonic : goc_test::bitfield_names)
    supports_dpp |= std::strcmp(name + sizeof("goc_") - 1, mnemonic) == 0;
  for (auto mnemonic : goc_test::integer_add_names)
    supports_dpp |= std::strcmp(name + sizeof("goc_") - 1, mnemonic) == 0;
  for (auto mnemonic : goc_test::dpp_integer_mul_names)
    supports_dpp |= std::strcmp(name + sizeof("goc_") - 1, mnemonic) == 0;
  for (auto mnemonic : goc_test::integer_mad_names)
    supports_dpp |= std::strcmp(name + sizeof("goc_") - 1, mnemonic) == 0;
  for (auto mnemonic : goc_test::dpp_boolean16_names)
    supports_dpp |= std::strcmp(name + sizeof("goc_") - 1, mnemonic) == 0;
  for (auto mnemonic : goc_test::integer16_names)
    supports_dpp |= std::strcmp(name + sizeof("goc_") - 1, mnemonic) == 0;
  for (auto mnemonic : goc_test::integer16_ternary_names)
    supports_dpp |= std::strcmp(name + sizeof("goc_") - 1, mnemonic) == 0;
  for (auto mnemonic : goc_test::half_binary_names)
    supports_dpp |= std::strcmp(name + sizeof("goc_") - 1, mnemonic) == 0;
  for (auto mnemonic : goc_test::half_minmax_names)
    supports_dpp |= std::strcmp(name + sizeof("goc_") - 1, mnemonic) == 0;
  for (auto mnemonic : goc_test::half_unary_names)
    supports_dpp |= std::strcmp(name + sizeof("goc_") - 1, mnemonic) == 0;
  for (auto mnemonic : goc_test::conversion32_names)
    supports_dpp |= std::strcmp(name + sizeof("goc_") - 1, mnemonic) == 0;
  for (auto mnemonic : goc_test::conversion16_names)
    supports_dpp |= std::strcmp(name + sizeof("goc_") - 1, mnemonic) == 0;
  for (auto mnemonic : {"v_cvt_f32_ubyte0", "v_cvt_f32_ubyte1", "v_cvt_f32_ubyte2",
                        "v_cvt_f32_ubyte3", "v_cvt_off_f32_i4", "v_cvt_pk_u8_f32", "v_cvt_i32_i16",
                        "v_cvt_u32_u16", "v_cvt_pk_i16_i32", "v_cvt_pk_u16_u32"})
    supports_dpp |= std::strcmp(name + sizeof("goc_") - 1, mnemonic) == 0;
  for (auto mnemonic : {"v_cvt_pk_norm_i16_f32", "v_cvt_pk_norm_u16_f32", "v_cvt_pk_norm_i16_f16",
                        "v_cvt_pk_norm_u16_f16", "v_cvt_norm_i16_f16", "v_cvt_norm_u16_f16"})
    supports_dpp |= std::strcmp(name + sizeof("goc_") - 1, mnemonic) == 0;
  for (auto mnemonic : {"v_cvt_pk_rtz_f16_f32", "v_cvt_pk_i16_f32", "v_cvt_pk_u16_f32"})
    supports_dpp |= std::strcmp(name + sizeof("goc_") - 1, mnemonic) == 0;
  for (auto mnemonic :
       {"v_cvt_pk_fp8_f32", "v_cvt_pk_bf8_f32", "v_cvt_sr_fp8_f32", "v_cvt_sr_bf8_f32"})
    supports_dpp |= std::strcmp(name + sizeof("goc_") - 1, mnemonic) == 0;
  for (auto mnemonic : {"v_cvt_f32_fp8", "v_cvt_f32_bf8"})
    supports_dpp |= std::strcmp(name + sizeof("goc_") - 1, mnemonic) == 0;
  for (auto mnemonic : {"v_sad_u8", "v_sad_hi_u8", "v_sad_u16", "v_sad_u32", "v_msad_u8"})
    supports_dpp |= std::strcmp(name + sizeof("goc_") - 1, mnemonic) == 0;
  for (auto mnemonic : {"v_dot2_f16_f16", "v_dot2_bf16_bf16"})
    supports_dpp |= std::strcmp(name + sizeof("goc_") - 1, mnemonic) == 0;
  for (auto mnemonic :
       {"v_dot2_f32_f16", "v_dot2_f32_bf16", "v_dot2_f32_f16_rdna4", "v_dot2_f32_bf16_rdna4"})
    supports_dpp |= std::strcmp(name + sizeof("goc_") - 1, mnemonic) == 0;
  for (auto mnemonic : {"v_sat_pk_u8_i16", "v_pack_b32_f16"})
    supports_dpp |= std::strcmp(name + sizeof("goc_") - 1, mnemonic) == 0;
  for (auto mnemonic : {"v_cndmask_b32", "v_cndmask_b16"})
    supports_dpp |= std::strcmp(name + sizeof("goc_") - 1, mnemonic) == 0;
  for (auto mnemonic : {"v_sin_f32", "v_cos_f32", "v_sin_f16", "v_cos_f16"})
    supports_dpp |= std::strcmp(name + sizeof("goc_") - 1, mnemonic) == 0;
  for (auto mnemonic : {"v_cubeid_f32", "v_cubesc_f32", "v_cubetc_f32", "v_cubema_f32"})
    supports_dpp |= std::strcmp(name + sizeof("goc_") - 1, mnemonic) == 0;
  for (auto mnemonic : {"v_fma_mix_f32", "v_fma_mixlo_f16", "v_fma_mixhi_f16",
                        "v_fma_mix_f32_rdna4", "v_fma_mixlo_f16_rdna4", "v_fma_mixhi_f16_rdna4"})
    supports_dpp |= std::strcmp(name + sizeof("goc_") - 1, mnemonic) == 0;
  supports_dpp |= std::strcmp(name, "goc_v_mullit_f32") == 0;
  supports_dpp |= std::strcmp(name, "goc_v_div_fixup_f16") == 0;
  supports_dpp |= std::strcmp(name, "goc_v_fma_dx9_zero_f32") == 0 ||
                  std::strcmp(name, "goc_v_fma_dx9_zero_f32_rdna4") == 0;
  supports_dpp |= std::strcmp(name, "goc_v_fmac_dx9_zero_f32") == 0;
  for (auto mnemonic : {"goc_v_min_f16", "goc_v_max_f16", "goc_v_min_f32", "goc_v_max_f32",
                        "goc_v_min3_f16", "goc_v_max3_f16", "goc_v_minmax_f16", "goc_v_maxmin_f16",
                        "goc_v_med3_f16", "goc_v_min3_f32", "goc_v_max3_f32", "goc_v_minmax_f32",
                        "goc_v_maxmin_f32", "goc_v_med3_f32"})
    supports_dpp |= std::strcmp(name, mnemonic) == 0;
  for (auto mnemonic :
       {"goc_v_frexp_mant_f32_rdna4", "goc_v_trunc_f32_rdna4", "goc_v_ceil_f32_rdna4",
        "goc_v_rndne_f32_rdna4", "goc_v_floor_f32_rdna4", "goc_v_sqrt_f32_rdna4",
        "goc_v_rcp_f32_rdna4", "goc_v_rsq_f32_rdna4", "goc_v_exp_f32_rdna4", "goc_v_log_f32_rdna4"})
    supports_dpp |= std::strcmp(name, mnemonic) == 0;
  bool dpp_scalar_output = std::strcmp(name, "goc_v_rcp_iflag_f32") == 0 ||
                           std::strcmp(name, "goc_v_rcp_iflag_f32_rdna4") == 0;
  for (auto mnemonic : {"v_add_co_u32", "v_sub_co_u32", "v_subrev_co_u32", "v_add_co_ci_u32",
                        "v_sub_co_ci_u32", "v_subrev_co_ci_u32"})
    dpp_scalar_output |= std::strcmp(name + sizeof("goc_") - 1, mnemonic) == 0;
  for (unsigned op = 0; op < 48; ++op)
    dpp_scalar_output |=
        std::strcmp(name + sizeof("goc_") - 1, goc_test::integer_compare_names[op]) == 0;
  for (unsigned op = 0; op < 56; ++op)
    dpp_scalar_output |=
        std::strcmp(name + sizeof("goc_") - 1, goc_test::float_compare_names[op]) == 0;
  for (auto mnemonic :
       {"v_cmp_class_f16", "v_cmpx_class_f16", "v_cmp_class_f32", "v_cmpx_class_f32"})
    dpp_scalar_output |= std::strcmp(name + sizeof("goc_") - 1, mnemonic) == 0;
  for (auto mnemonic :
       {"v_cmp_f_f16", "v_cmp_t_f16", "v_cmpx_f_f16", "v_cmpx_t_f16", "v_cmp_f_f32", "v_cmp_t_f32",
        "v_cmpx_f_f32", "v_cmpx_t_f32", "v_cmp_f_i32", "v_cmp_t_i32", "v_cmp_f_u32", "v_cmp_t_u32",
        "v_cmpx_f_i32", "v_cmpx_t_i32", "v_cmpx_f_u32", "v_cmpx_t_u32"})
    dpp_scalar_output |= std::strcmp(name + sizeof("goc_") - 1, mnemonic) == 0;
  supports_dpp |= dpp_scalar_output;
  for (unsigned bit = 32; bit < 64; ++bit)
    for (Mask exec_mask : {Mask(0), Mask(~Mask(0))}) {
      // A DPP enable bit alone is a valid descriptor (zero fields).
      if (supports_dpp && (bit == 32 || bit == 34)) {
        // Family tests supply required scalar outputs even for zero EXEC.
        if (exec_mask == 0 && !dpp_scalar_output) {
          EXPECT_EQ(instruction(0, exec_mask, 1ULL << bit, Operands{}...), GOC_SUCCESS);
        }
        continue;
      }
      // Invalid flags must be rejected before reading even required scalar outputs.
      EXPECT_EQ(instruction(0, exec_mask, 1ULL << bit, Operands{}...), GOC_ERROR_INVALID_FLAGS)
          << bit;
    }
}

} // namespace

TEST(InstructionFlags, EveryEntryPointRejectsUnsupportedHighBits) {
#define GOC_API(name) check_high_flags(#name, name);
#include "api_symbols.inc"
#undef GOC_API
}

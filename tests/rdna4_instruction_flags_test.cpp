// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_bitfield_reference.h"
#include "rdna4_boolean_reference.h"
#include "rdna4_conversion16_reference.h"
#include "rdna4_conversion32_reference.h"
#include "rdna4_dpp_arithmetic_reference.h"
#include "rdna4_dpp_integer_reference.h"
#include "rdna4_half_binary_reference.h"
#include "rdna4_half_minmax_reference.h"
#include "rdna4_half_unary_reference.h"
#include "rdna4_integer16_reference.h"
#include "rdna4_integer16_ternary_reference.h"
#include "rdna4_integer_add_reference.h"
#include "rdna4_integer_mad_reference.h"
#include "rdna4_integer_minmax_reference.h"
#include "rdna4_integer_mul_reference.h"
#include "rdna4_integer_ternary_reference.h"
#include "rdna4_unary_reference.h"

#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>

namespace {

template <typename... Operands>
void check_high_flags(const char *name,
                      int (*instruction)(uint64_t, uint64_t, uint64_t, Operands...)) {
  SCOPED_TRACE(name);
  bool supports_dpp = std::strcmp(name, "goc_rdna4_v_fma_f32") == 0 ||
                      std::strcmp(name, "goc_rdna4_v_fmac_f32") == 0 ||
                      std::strcmp(name, "goc_rdna4_v_fma_f16") == 0 ||
                      std::strcmp(name, "goc_rdna4_v_fmac_f16") == 0 ||
                      std::strcmp(name, "goc_rdna4_v_ldexp_f16") == 0 ||
                      std::strcmp(name, "goc_rdna4_v_frexp_exp_i16_f16") == 0 ||
                      std::strcmp(name, "goc_rdna4_v_frexp_exp_i32_f32") == 0 ||
                      std::strcmp(name, "goc_rdna4_v_ldexp_f32") == 0;
  for (auto mnemonic : goc_test::dpp_arithmetic_names)
    supports_dpp |= std::strcmp(name + sizeof("goc_rdna4_") - 1, mnemonic) == 0;
  for (auto mnemonic : goc_test::unary_names)
    supports_dpp |= std::strcmp(name + sizeof("goc_rdna4_") - 1, mnemonic) == 0;
  for (auto mnemonic : goc_test::dpp_integer_names)
    supports_dpp |= std::strcmp(name + sizeof("goc_rdna4_") - 1, mnemonic) == 0;
  for (auto mnemonic : goc_test::integer_minmax_names)
    supports_dpp |= std::strcmp(name + sizeof("goc_rdna4_") - 1, mnemonic) == 0;
  for (auto mnemonic : goc_test::integer_ternary_names)
    supports_dpp |= std::strcmp(name + sizeof("goc_rdna4_") - 1, mnemonic) == 0;
  for (auto mnemonic : goc_test::bitfield_names)
    supports_dpp |= std::strcmp(name + sizeof("goc_rdna4_") - 1, mnemonic) == 0;
  for (auto mnemonic : goc_test::integer_add_names)
    supports_dpp |= std::strcmp(name + sizeof("goc_rdna4_") - 1, mnemonic) == 0;
  for (auto mnemonic : goc_test::dpp_integer_mul_names)
    supports_dpp |= std::strcmp(name + sizeof("goc_rdna4_") - 1, mnemonic) == 0;
  for (auto mnemonic : goc_test::integer_mad_names)
    supports_dpp |= std::strcmp(name + sizeof("goc_rdna4_") - 1, mnemonic) == 0;
  for (auto mnemonic : goc_test::dpp_boolean16_names)
    supports_dpp |= std::strcmp(name + sizeof("goc_rdna4_") - 1, mnemonic) == 0;
  for (auto mnemonic : goc_test::integer16_names)
    supports_dpp |= std::strcmp(name + sizeof("goc_rdna4_") - 1, mnemonic) == 0;
  for (auto mnemonic : goc_test::integer16_ternary_names)
    supports_dpp |= std::strcmp(name + sizeof("goc_rdna4_") - 1, mnemonic) == 0;
  for (auto mnemonic : goc_test::half_binary_names)
    supports_dpp |= std::strcmp(name + sizeof("goc_rdna4_") - 1, mnemonic) == 0;
  for (auto mnemonic : goc_test::half_minmax_names)
    supports_dpp |= std::strcmp(name + sizeof("goc_rdna4_") - 1, mnemonic) == 0;
  for (auto mnemonic : goc_test::half_unary_names)
    supports_dpp |= std::strcmp(name + sizeof("goc_rdna4_") - 1, mnemonic) == 0;
  for (auto mnemonic : goc_test::conversion32_names)
    supports_dpp |= std::strcmp(name + sizeof("goc_rdna4_") - 1, mnemonic) == 0;
  for (auto mnemonic : goc_test::conversion16_names)
    supports_dpp |= std::strcmp(name + sizeof("goc_rdna4_") - 1, mnemonic) == 0;
  for (auto mnemonic : {"v_cvt_f32_ubyte0", "v_cvt_f32_ubyte1", "v_cvt_f32_ubyte2",
                        "v_cvt_f32_ubyte3", "v_cvt_off_f32_i4", "v_cvt_pk_u8_f32", "v_cvt_i32_i16",
                        "v_cvt_u32_u16", "v_cvt_pk_i16_i32", "v_cvt_pk_u16_u32"})
    supports_dpp |= std::strcmp(name + sizeof("goc_rdna4_") - 1, mnemonic) == 0;
  for (auto mnemonic : {"v_cvt_pk_norm_i16_f32", "v_cvt_pk_norm_u16_f32", "v_cvt_pk_norm_i16_f16",
                        "v_cvt_pk_norm_u16_f16", "v_cvt_norm_i16_f16", "v_cvt_norm_u16_f16"})
    supports_dpp |= std::strcmp(name + sizeof("goc_rdna4_") - 1, mnemonic) == 0;
  for (auto mnemonic : {"v_cvt_pk_rtz_f16_f32", "v_cvt_pk_i16_f32", "v_cvt_pk_u16_f32"})
    supports_dpp |= std::strcmp(name + sizeof("goc_rdna4_") - 1, mnemonic) == 0;
  for (auto mnemonic :
       {"v_cvt_pk_fp8_f32", "v_cvt_pk_bf8_f32", "v_cvt_sr_fp8_f32", "v_cvt_sr_bf8_f32"})
    supports_dpp |= std::strcmp(name + sizeof("goc_rdna4_") - 1, mnemonic) == 0;
  for (auto mnemonic : {"v_cvt_f32_fp8", "v_cvt_f32_bf8"})
    supports_dpp |= std::strcmp(name + sizeof("goc_rdna4_") - 1, mnemonic) == 0;
  for (auto mnemonic : {"v_sad_u8", "v_sad_hi_u8", "v_sad_u16", "v_sad_u32", "v_msad_u8"})
    supports_dpp |= std::strcmp(name + sizeof("goc_rdna4_") - 1, mnemonic) == 0;
  for (auto mnemonic : {"v_dot2_f16_f16", "v_dot2_bf16_bf16"})
    supports_dpp |= std::strcmp(name + sizeof("goc_rdna4_") - 1, mnemonic) == 0;
  for (auto mnemonic : {"v_dot2_f32_f16", "v_dot2_f32_bf16"})
    supports_dpp |= std::strcmp(name + sizeof("goc_rdna4_") - 1, mnemonic) == 0;
  for (auto mnemonic : {"v_sat_pk_u8_i16", "v_pack_b32_f16"})
    supports_dpp |= std::strcmp(name + sizeof("goc_rdna4_") - 1, mnemonic) == 0;
  for (auto mnemonic : {"v_cndmask_b32", "v_cndmask_b16"})
    supports_dpp |= std::strcmp(name + sizeof("goc_rdna4_") - 1, mnemonic) == 0;
  for (auto mnemonic : {"v_sin_f32", "v_cos_f32", "v_sin_f16", "v_cos_f16"})
    supports_dpp |= std::strcmp(name + sizeof("goc_rdna4_") - 1, mnemonic) == 0;
  for (auto mnemonic : {"v_cubeid_f32", "v_cubesc_f32", "v_cubetc_f32", "v_cubema_f32"})
    supports_dpp |= std::strcmp(name + sizeof("goc_rdna4_") - 1, mnemonic) == 0;
  bool dpp_scalar_output = std::strcmp(name, "goc_rdna4_v_rcp_iflag_f32") == 0;
  for (auto mnemonic : {"v_add_co_u32", "v_sub_co_u32", "v_subrev_co_u32", "v_add_co_ci_u32",
                        "v_sub_co_ci_u32", "v_subrev_co_ci_u32"})
    dpp_scalar_output |= std::strcmp(name + sizeof("goc_rdna4_") - 1, mnemonic) == 0;
  supports_dpp |= dpp_scalar_output;
  for (unsigned bit = 32; bit < 64; ++bit)
    for (uint64_t exec : {UINT64_C(0), UINT64_MAX}) {
      // A DPP enable bit alone is a valid descriptor (zero fields).
      if (supports_dpp && (bit == 32 || bit == 34)) {
        // Family tests supply required scalar outputs even for zero EXEC.
        if (exec == 0 && !dpp_scalar_output) {
          EXPECT_EQ(instruction(0, exec, UINT64_C(1) << bit, Operands{}...), GOC_SUCCESS);
        }
        continue;
      }
      // Invalid flags must be rejected before reading even required scalar outputs.
      EXPECT_EQ(instruction(0, exec, UINT64_C(1) << bit, Operands{}...), GOC_ERROR_INVALID_FLAGS)
          << bit;
    }
}

} // namespace

TEST(InstructionFlags, EveryEntryPointRejectsUnsupportedHighBits) {
#define GOC_RDNA4_API(name) check_high_flags(#name, name);
#include "rdna4_api_symbols.inc"
#undef GOC_RDNA4_API
}

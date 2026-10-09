// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_bitfield_reference.h"
#include "rdna4_boolean_reference.h"
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
  for (unsigned bit = 32; bit < 64; ++bit)
    for (uint64_t exec : {UINT64_C(0), UINT64_MAX}) {
      // A DPP enable bit alone is a valid descriptor (zero fields).
      if (supports_dpp && (bit == 32 || bit == 34)) {
        if (exec == 0) {
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

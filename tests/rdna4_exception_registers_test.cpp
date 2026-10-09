// SPDX-License-Identifier: MIT

#include "goc/goc.h"

#include <gtest/gtest.h>
#include <stdint.h>
#include <tuple>
#include <type_traits>
#include <utility>

namespace {

template <typename... Args, size_t... I>
void check_unsupported(int (*instruction)(Args...), std::index_sequence<I...>) {
  using Operands = std::tuple<Args...>;
  static_assert(std::is_same_v<std::tuple_element_t<sizeof...(Args) - 1, Operands>, uint32_t *>);
  uint32_t excp_flag_user = 0xa5000041;
  // Null operands deliberately verify rejection before any operand access.
  // The sentinel includes prior exceptions and unrelated/reserved bits.
  EXPECT_EQ(instruction(std::tuple_element_t<I, Operands>{}..., &excp_flag_user),
            GOC_ERROR_UNSUPPORTED_EXCEPTIONS);
  EXPECT_EQ(excp_flag_user, 0xa5000041U);
}

template <typename... Args> void check_unsupported(int (*instruction)(Args...)) {
  check_unsupported(instruction, std::make_index_sequence<sizeof...(Args) - 1>{});
}

} // namespace

TEST(ExceptionRegisters, UnimplementedReportingRejectsBeforeOperandAccess) {
#define GOC_RDNA4_REPORTING_API(symbol)                                                            \
  {                                                                                                \
    SCOPED_TRACE(#symbol);                                                                         \
    check_unsupported(symbol);                                                                     \
  }
#include "rdna4_reporting_symbols.inc"
#undef GOC_RDNA4_REPORTING_API
}

TEST(ExceptionRegisters, UnsupportedLeavesAllOutputsUntouched) {
  uint32_t data[32] = {0xdeadbeef};
  uint32_t *d[] = {data};
  uint32_t excp_flag_user = 0x80000021;
  EXPECT_EQ(goc_rdna4_v_fma_f32(0, UINT32_MAX, 0, d, nullptr, nullptr, nullptr, &excp_flag_user),
            GOC_ERROR_UNSUPPORTED_EXCEPTIONS);
  EXPECT_EQ(data[0], 0xdeadbeefU);
  EXPECT_EQ(excp_flag_user, 0x80000021U);
  uint32_t scc = 1;
  EXPECT_EQ(goc_rdna4_s_cmp_eq_f32(0, 0, 0, &scc, 0, 0, &excp_flag_user),
            GOC_ERROR_UNSUPPORTED_EXCEPTIONS);
  EXPECT_EQ(scc, 1U);
  EXPECT_EQ(excp_flag_user, 0x80000021U);
  // Opting out keeps validation and zero-EXEC behavior unchanged.
  EXPECT_EQ(goc_rdna4_v_fma_f32(0, 0, 0, nullptr, nullptr, nullptr, nullptr, nullptr), GOC_SUCCESS);
  EXPECT_EQ(goc_rdna4_v_fma_f32(0, 0, 1ULL << 63, nullptr, nullptr, nullptr, nullptr, nullptr),
            GOC_ERROR_INVALID_FLAGS);
  // Reporting rejection takes precedence over other unsupported options.
  EXPECT_EQ(goc_rdna4_v_fma_f32(UINT64_MAX, 0, UINT64_MAX, nullptr, nullptr, nullptr, nullptr,
                                &excp_flag_user),
            GOC_ERROR_UNSUPPORTED_EXCEPTIONS);
  EXPECT_EQ(excp_flag_user, 0x80000021U);
}

TEST(ExceptionRegisters, NonReportingInstructionsKeepTheirSignatures) {
  using Unary = int (*)(uint64_t, uint32_t, uint64_t, uint32_t *const *, const uint32_t *const *);
  using Ternary = int (*)(uint64_t, uint32_t, uint64_t, uint32_t *const *, const uint32_t *const *,
                          const uint32_t *const *, const uint32_t *const *);
  static_assert(std::is_same_v<decltype(&goc_rdna4_v_cvt_f32_ubyte0), Unary>);
  static_assert(std::is_same_v<decltype(&goc_rdna4_v_wmma_f32_16x16x16_f16), Ternary>);
  static_assert(std::is_same_v<decltype(&goc_rdna4_v_dot4_f32_fp8_fp8), Ternary>);
  static_assert(std::is_same_v<decltype(&goc_rdna4_v_interp_p10_f32), Ternary>);
}

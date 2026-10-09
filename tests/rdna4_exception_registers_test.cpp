// SPDX-License-Identifier: MIT

#include "goc/goc.h"

#include <gtest/gtest.h>
#include <initializer_list>
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
  for (uint64_t strict : std::initializer_list<uint64_t>{0, GOC_SEMANTICS_STRICT})
    EXPECT_EQ(instruction(GOC_SEMANTICS_EXACT_EMPIRICAL | strict,
                          std::tuple_element_t<I + 1, Operands>{}..., &excp_flag_user),
              GOC_ERROR_UNSUPPORTED_EXCEPTIONS);
  EXPECT_EQ(excp_flag_user, 0xa5000041U);
}

template <typename... Args> void check_unsupported(int (*instruction)(Args...)) {
  check_unsupported(instruction, std::make_index_sequence<sizeof...(Args) - 2>{});
}

// Scalar instructions execute even with empty EXEC, so provide their outputs.
template <typename T> T empty_operand(uint32_t &word, uint64_t &wide_word) {
  if constexpr (std::is_same_v<T, uint32_t *>)
    return &word;
  else if constexpr (std::is_same_v<T, uint64_t *>)
    return &wide_word;
  else
    return T{};
}

template <typename... Args, size_t... I>
void check_loose(int (*instruction)(Args...), std::index_sequence<I...>) {
  using Operands = std::tuple<Args...>;
  uint32_t word = 0, excp_flag_user = 0xa5000041;
  uint64_t wide_word = 0;
  EXPECT_EQ(instruction(empty_operand<std::tuple_element_t<I, Operands>>(word, wide_word)...,
                        &excp_flag_user),
            GOC_SUCCESS);
  EXPECT_EQ(excp_flag_user, 0xa5000041U);
}

template <typename... Args> void check_loose(int (*instruction)(Args...)) {
  check_loose(instruction, std::make_index_sequence<sizeof...(Args) - 1>{});
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

TEST(ExceptionRegisters, LooseAllowsUnimplementedReporting) {
#define GOC_RDNA4_REPORTING_API(symbol)                                                            \
  {                                                                                                \
    SCOPED_TRACE(#symbol);                                                                         \
    check_loose(symbol);                                                                           \
  }
#include "rdna4_reporting_symbols.inc"
#undef GOC_RDNA4_REPORTING_API
}

TEST(ExceptionRegisters, LooseComputesResultsAndPreservesPriorFlags) {
  uint32_t a[32] = {0x40000000}, b[32] = {0x40400000}, c[32] = {0x40800000};
  const uint32_t *ap[] = {a}, *bp[] = {b}, *cp[] = {c};
  uint32_t data[32] = {}, *d[] = {data};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint32_t excp_flag_user = 0x80000021;
    EXPECT_EQ(goc_rdna4_v_fma_f32(cpu, 1, 0, d, ap, bp, cp, &excp_flag_user), GOC_SUCCESS);
    EXPECT_EQ(data[0], 0x41200000U);
    EXPECT_EQ(excp_flag_user, 0x80000021U);
    EXPECT_EQ(goc_rdna4_v_fma_f32(cpu, 1, 1ULL << 63, d, ap, bp, cp, &excp_flag_user),
              GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(data[0], 0x41200000U);
    EXPECT_EQ(excp_flag_user, 0x80000021U);
  }
}

TEST(ExceptionRegisters, UnsupportedLeavesAllOutputsUntouched) {
  uint32_t data[32] = {0xdeadbeef};
  uint32_t *d[] = {data};
  uint32_t excp_flag_user = 0x80000021;
  EXPECT_EQ(goc_rdna4_v_fma_f32(GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, 0, d, nullptr, nullptr,
                                nullptr, &excp_flag_user),
            GOC_ERROR_UNSUPPORTED_EXCEPTIONS);
  EXPECT_EQ(data[0], 0xdeadbeefU);
  EXPECT_EQ(excp_flag_user, 0x80000021U);
  uint32_t scc = 1;
  EXPECT_EQ(
      goc_rdna4_s_cmp_eq_f32(GOC_SEMANTICS_EXACT_EMPIRICAL, 0, 0, &scc, 0, 0, &excp_flag_user),
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

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
              GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
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

// All operands are null/zero: unsupported reporting must reject before DPP
// routing or accumulator access, even with nonzero EXEC and invalid modifiers.
template <typename... Args, size_t... I>
void check_half_fma_reporting(int (*instruction)(Args...), bool rdna3, std::index_sequence<I...>) {
  using Operands = std::tuple<Args...>;
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (uint64_t semantics : {GOC_SEMANTICS_EXACT_EMPIRICAL, uint64_t(2) << 16, uint64_t(3) << 16})
      for (uint64_t strict : {uint64_t(0), GOC_SEMANTICS_STRICT})
        for (unsigned policy = 0; policy < 4; ++policy) {
          uint64_t flags = cpu | semantics | strict |
                           (policy & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                           (policy & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0);
          if (!rdna3 && policy == 0 && semantics == GOC_SEMANTICS_EXACT_EMPIRICAL)
            continue; // This reporting policy is implemented and needs real operands.
          for (uint64_t exec_mask : {0ULL, ~0ULL})
            for (uint64_t mode : {uint64_t(0), GOC_DPP8, uint64_t(1) << 31, uint64_t(1) << 63}) {
              uint32_t state = 0xa5000040;
              EXPECT_EQ(instruction(flags, std::tuple_element_t<1, Operands>(exec_mask), mode,
                                    std::tuple_element_t<I + 3, Operands>{}..., &state),
                        GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
              EXPECT_EQ(state, 0xa5000040U);
            }
        }
}

template <typename... Args> void check_half_fma_reporting(int (*instruction)(Args...), bool rdna3) {
  check_half_fma_reporting(instruction, rdna3, std::make_index_sequence<sizeof...(Args) - 4>{});
}

} // namespace

TEST(ExceptionRegisters, UnimplementedReportingRejectsBeforeOperandAccess) {
#define GOC_REPORTING_API(symbol)                                                                  \
  {                                                                                                \
    SCOPED_TRACE(#symbol);                                                                         \
    check_unsupported(symbol);                                                                     \
  }
#include "reporting_symbols.inc"
#undef GOC_REPORTING_API
}

TEST(ExceptionRegisters, LooseAllowsUnimplementedReporting) {
#define GOC_REPORTING_API(symbol)                                                                  \
  {                                                                                                \
    SCOPED_TRACE(#symbol);                                                                         \
    check_loose(symbol);                                                                           \
  }
#include "reporting_symbols.inc"
#undef GOC_REPORTING_API
}

TEST(ExceptionRegisters, LooseComputesResultsAndPreservesPriorFlags) {
  uint32_t a[32] = {0x40000000}, b[32] = {0x40400000}, c[32] = {0x40800000};
  const uint32_t *ap[] = {a}, *bp[] = {b}, *cp[] = {c};
  uint32_t data[32] = {}, *d[] = {data};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint32_t excp_flag_user = 0x80000021;
    EXPECT_EQ(goc_v_fma_f32_rdna4(cpu, 1, 0, d, ap, bp, cp, &excp_flag_user), GOC_SUCCESS);
    EXPECT_EQ(data[0], 0x41200000U);
    EXPECT_EQ(excp_flag_user, 0x80000021U);
    EXPECT_EQ(goc_v_fma_f32_rdna4(cpu, 1, 1ULL << 63, d, ap, bp, cp, &excp_flag_user),
              GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(data[0], 0x41200000U);
    EXPECT_EQ(excp_flag_user, 0x80000021U);
  }
}

TEST(ExceptionRegisters, UnsupportedLeavesAllOutputsUntouched) {
  uint32_t data[32] = {0xdeadbeef};
  uint32_t *d[] = {data};
  uint32_t state = 0x80000021;
  EXPECT_EQ(goc_v_fma_f32(GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, 0, d, nullptr, nullptr,
                          nullptr, &state),
            GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
  EXPECT_EQ(data[0], 0xdeadbeefU);
  EXPECT_EQ(state, 0x80000021U);
  uint32_t scc = 1;
  EXPECT_EQ(goc_s_add_f32(GOC_SEMANTICS_EXACT_EMPIRICAL, 0, &scc, 0, 0, &state),
            GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
  EXPECT_EQ(scc, 1U);
  EXPECT_EQ(state, 0x80000021U);
  // Opting out keeps validation and zero-EXEC behavior unchanged.
  EXPECT_EQ(goc_v_fma_f32_rdna4(0, 0, 0, nullptr, nullptr, nullptr, nullptr, nullptr), GOC_SUCCESS);
  EXPECT_EQ(goc_v_fma_f32_rdna4(0, 0, 1ULL << 63, nullptr, nullptr, nullptr, nullptr, nullptr),
            GOC_ERROR_INVALID_FLAGS);
  // Reporting rejection takes precedence over other unsupported options.
  EXPECT_EQ(
      goc_v_fma_f32_rdna4(UINT64_MAX, 0, UINT64_MAX, nullptr, nullptr, nullptr, nullptr, &state),
      GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
  EXPECT_EQ(state, 0x80000021U);
}

TEST(ExceptionRegisters, NonReportingInstructionsKeepTheirSignatures) {
  using Unary = int (*)(uint64_t, uint32_t, uint64_t, uint32_t *const *, const uint32_t *const *);
  using Ternary = int (*)(uint64_t, uint32_t, uint64_t, uint32_t *const *, const uint32_t *const *,
                          const uint32_t *const *, const uint32_t *const *);
  static_assert(std::is_same_v<decltype(&goc_v_cvt_f32_ubyte0_rdna4), Unary>);
  using Matrix = int (*)(uint64_t, uint64_t, uint32_t *const *, const uint32_t *const *,
                         const uint32_t *const *, const uint32_t *const *);
  static_assert(std::is_same_v<decltype(&goc_v_wmma_f32_16x16x16_f16_rdna4), Matrix>);
  static_assert(std::is_same_v<decltype(&goc_v_dot4_f32_fp8_fp8), Ternary>);
  static_assert(std::is_same_v<decltype(&goc_v_interp_p10_f32), Ternary>);
}

TEST(ExceptionRegisters, HalfFmaReportingPrecedesRoutingAndOperandAccess) {
  check_half_fma_reporting(goc_v_fma_f16, true);
  check_half_fma_reporting(goc_v_fma_f16_wave64, true);
  check_half_fma_reporting(goc_v_fma_f16_rdna4, false);
  check_half_fma_reporting(goc_v_fma_f16_rdna4_wave64, false);
  check_half_fma_reporting(goc_v_fmac_f16, true);
  check_half_fma_reporting(goc_v_fmac_f16_wave64, true);
  check_half_fma_reporting(goc_v_fmac_f16_rdna4, false);
  check_half_fma_reporting(goc_v_fmac_f16_rdna4_wave64, false);
  check_half_fma_reporting(goc_v_fmaak_f16, true);
  check_half_fma_reporting(goc_v_fmaak_f16_wave64, true);
  check_half_fma_reporting(goc_v_fmaak_f16_rdna4, false);
  check_half_fma_reporting(goc_v_fmaak_f16_rdna4_wave64, false);
  check_half_fma_reporting(goc_v_fmamk_f16, true);
  check_half_fma_reporting(goc_v_fmamk_f16_wave64, true);
  check_half_fma_reporting(goc_v_fmamk_f16_rdna4, false);
  check_half_fma_reporting(goc_v_fmamk_f16_rdna4_wave64, false);
  check_half_fma_reporting(goc_v_pk_fma_f16, true);
  check_half_fma_reporting(goc_v_pk_fma_f16_wave64, true);
  check_half_fma_reporting(goc_v_pk_fma_f16_rdna4, false);
  check_half_fma_reporting(goc_v_pk_fma_f16_rdna4_wave64, false);
  check_half_fma_reporting(goc_v_pk_fmac_f16, true);
  check_half_fma_reporting(goc_v_pk_fmac_f16_wave64, true);
  check_half_fma_reporting(goc_v_pk_fmac_f16_rdna4, false);
  check_half_fma_reporting(goc_v_pk_fmac_f16_rdna4_wave64, false);
}

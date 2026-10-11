// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "fp_environment.h"
#include "fuzzy.h"
#include "goc/goc.h"
#include "half_reference.h"
#include "half_unary_reference.h"
#include "loose_tolerances.h"
#include "pseudo_scalar_hardware.h"
#include "pseudo_scalar_reference.h"
#include "unary_checkpoints.h"
#include "unary_exact_hardware.h"

#include <cfenv>
#include <cmath>
#include <gtest/gtest.h>
#include <initializer_list>
#include <iomanip>
#include <stdint.h>

TEST(PseudoScalar, HardwareEveryModifierAndFpState) {
  const uint64_t max_cpu = goc_init_cpu_flags();
  for (unsigned state = 0; state < 8; ++state)
    for (unsigned op = 0; op < 10; ++op)
      for (unsigned m = 0; m < 32; ++m) {
        const auto &gold =
            goc_test::pseudo_scalar_outputs[goc_test::pseudo_scalar_blocks[(state * 10 + op) * 32 +
                                                                           m]];
        for (unsigned sample = 0; sample < 48; ++sample)
          for (uint64_t cpu = 0; cpu <= max_cpu; ++cpu) {
            uint32_t result = 0xdeadbeef, a = goc_test::pseudo_scalar_input(
                                              op & 1, goc_test::pseudo_scalar_samples[sample]);
            ASSERT_EQ(goc_test::pseudo_scalar_functions[op](
                          cpu | goc_test::pseudo_scalar_flags(state),
                          goc_test::pseudo_scalar_mode(m), &result, a, nullptr),
                      GOC_SUCCESS);
            ASSERT_TRUE(goc_test::pseudo_scalar_close(result, gold[sample], op & 1))
                << state << "/" << op << "/" << m << "/" << sample << "/" << cpu << std::hex
                << " result " << result << " expected " << gold[sample];
          }
      }
}

TEST(PseudoScalar, ScalarStorageMayAlias) {
  const uint64_t max_cpu = goc_init_cpu_flags();
  for (unsigned state = 0; state < 8; ++state)
    for (unsigned op = 0; op < 10; ++op)
      for (unsigned m = 0; m < 32; ++m) {
        const auto &gold =
            goc_test::pseudo_scalar_outputs[goc_test::pseudo_scalar_blocks[(state * 10 + op) * 32 +
                                                                           m]];
        for (unsigned sample : {0u, 1u, 2u, 9u, 16u, 20u, 42u})
          for (uint64_t cpu = 0; cpu <= max_cpu; ++cpu) {
            uint32_t words[] = {
                0x12345678,
                goc_test::pseudo_scalar_input(op & 1, goc_test::pseudo_scalar_samples[sample]),
                0x87654321};
            ASSERT_EQ(goc_test::pseudo_scalar_functions[op](
                          cpu | goc_test::pseudo_scalar_flags(state),
                          goc_test::pseudo_scalar_mode(m), words + 1, words[1], nullptr),
                      GOC_SUCCESS);
            ASSERT_TRUE(goc_test::pseudo_scalar_close(words[1], gold[sample], op & 1))
                << state << "/" << op << "/" << m << "/" << sample << "/" << cpu;
            ASSERT_EQ(words[0], 0x12345678u);
            ASSERT_EQ(words[2], 0x87654321u);
          }
      }
}

TEST(PseudoScalar, DenormalStagesZerosAndOverflow) {
  uint32_t d;
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (uint64_t semantics : {uint64_t(0), GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT}) {
      const uint64_t flags = cpu | semantics;
      ASSERT_EQ(goc_v_s_sqrt_f16(flags, 0, &d, 0xabcd0001, nullptr), GOC_SUCCESS);
      EXPECT_EQ(d, 0x0c00u);
      ASSERT_EQ(goc_v_s_sqrt_f16(flags | GOC_FP_FLUSH_INPUT_DENORMALS, 0, &d, 1, nullptr),
                GOC_SUCCESS);
      EXPECT_EQ(d, 0u);
      ASSERT_EQ(goc_v_s_rcp_f16(flags, 0, &d, 0x7bff, nullptr), GOC_SUCCESS);
      EXPECT_EQ(d, 0x0100u);
      ASSERT_EQ(goc_v_s_rcp_f16(flags | GOC_FP_FLUSH_OUTPUT_DENORMALS, 0, &d, 0x7bff, nullptr),
                GOC_SUCCESS);
      EXPECT_EQ(d, 0u);
      ASSERT_EQ(goc_v_s_rcp_f16(flags, GOC_ALU_OMOD_HALF, &d, 0xf00f, nullptr), GOC_SUCCESS);
      EXPECT_EQ(d, 0x8000u);
      ASSERT_EQ(goc_v_s_sqrt_f16(flags, GOC_ALU_OMOD_HALF, &d, 0x8000, nullptr), GOC_SUCCESS);
      EXPECT_EQ(d, 0u);
      ASSERT_EQ(goc_v_s_rcp_f16(flags | GOC_FP16_OVFL, 0, &d, 0x8000, nullptr), GOC_SUCCESS);
      EXPECT_EQ(d, 0xfbffu);
      ASSERT_EQ(goc_v_s_exp_f16(cpu | GOC_FP16_OVFL, 0, &d, 0x7c00, nullptr), GOC_SUCCESS);
      EXPECT_EQ(d, 0x7c00u);
      ASSERT_EQ(goc_v_s_rcp_f32(flags, 0, &d, 1, nullptr), GOC_SUCCESS);
      EXPECT_EQ(d, 0x7f800000u);
      ASSERT_EQ(goc_v_s_rcp_f32(flags, 0, &d, 0x7f7fffff, nullptr), GOC_SUCCESS);
      EXPECT_EQ(d, 0u);
    }
}

TEST(PseudoScalar, ValidationSemanticsAndHostRounding) {
  const uint32_t known = goc_test::pseudo_scalar_mode(31);
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  std::fesetround(FE_TONEAREST);
  for (unsigned op = 0; op < 10; ++op) {
    uint32_t d = 0xdeadbeef;
    for (unsigned bit = 0; bit < 32; ++bit)
      if (!(known & (1u << bit))) {
        EXPECT_EQ(goc_test::pseudo_scalar_functions[op](0, 1u << bit, &d, 0, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(d, 0xdeadbeef);
      }
    if (op == 1 || op == 3) {
      EXPECT_EQ(goc_test::pseudo_scalar_functions[op](
                    GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, known, &d, 0, nullptr),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
      EXPECT_EQ(d, 0xdeadbeef);
    }
    EXPECT_EQ(goc_test::pseudo_scalar_functions[op](1ULL << 63, known, &d, 0, nullptr),
              GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(d, 0xdeadbeef);
    EXPECT_EQ(goc_test::pseudo_scalar_functions[op](GOC_SEMANTICS_EXACT_EMPIRICAL, known, &d,
                                                    goc_test::pseudo_scalar_input(op & 1, 8),
                                                    nullptr),
              GOC_SUCCESS);
    EXPECT_EQ(std::fegetround(), FE_TONEAREST);
  }
}

TEST(PseudoScalar, ExactHardwareEveryModifierAndFpState) {
  const uint64_t exact = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;
  for (unsigned state = 0; state < 8; ++state)
    for (unsigned op : {0U, 2U, 4U, 5U, 6U, 7U, 8U, 9U})
      for (unsigned m = 0; m < 32; ++m) {
        const auto &gold =
            goc_test::pseudo_scalar_outputs[goc_test::pseudo_scalar_blocks[(state * 10 + op) * 32 +
                                                                           m]];
        for (unsigned sample = 0; sample < 48; ++sample)
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
            uint32_t words[] = {
                0x12345678,
                goc_test::pseudo_scalar_input(op & 1, goc_test::pseudo_scalar_samples[sample]),
                0x87654321};
            ASSERT_EQ(goc_test::pseudo_scalar_functions[op](
                          exact | cpu | goc_test::pseudo_scalar_flags(state),
                          goc_test::pseudo_scalar_mode(m), words + 1, words[1], nullptr),
                      GOC_SUCCESS);
            ASSERT_EQ(words[1], gold[sample]) << state << "/" << op << "/" << m << "/" << sample;
            ASSERT_EQ(words[0], 0x12345678U);
            ASSERT_EQ(words[2], 0x87654321U);
          }
      }
}

TEST(PseudoScalar, ExactGlobalStateRemainsExplicitlyUnsupported) {
  for (auto fn : goc_test::pseudo_scalar_functions)
    for (uint64_t strict : {uint64_t(0), GOC_SEMANTICS_STRICT})
      for (uint64_t mode : {uint64_t(0), uint64_t(GOC_ALU_NEG_B), uint64_t(1) << 63}) {
        uint32_t output = 0xdeadbeef, state = 0x12345678;
        EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | strict, mode, &output, 0, &state),
                  GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
        EXPECT_EQ(output, 0xdeadbeefU);
        EXPECT_EQ(state, 0x12345678U);
      }
}

TEST(PseudoScalar, ExactFp32BorrowedDenseDomains) {
  const goc_test::PseudoScalarFn functions[] = {goc_v_s_rcp_f32, goc_v_s_sqrt_f32, goc_v_s_rsq_f32,
                                                goc_v_s_exp_f32, goc_v_s_log_f32};
  const uint64_t exact = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;
  for (unsigned op = 0; op < 5; ++op) {
    uint64_t hash = goc_test::capture_hash_seed;
    for (unsigned index = 0; index < (1U << (op == 0 ? 23 : op == 3 ? 25 : 24)); ++index) {
      uint32_t input = op == 3   ? (0x3f000000 + (index & 0xffffff)) | ((index >> 24) << 31)
                       : op == 4 ? 0x3f000000 + index
                                 : 0x3f800000 + index;
      uint32_t result;
      ASSERT_EQ(functions[op](exact, 0, &result, input, nullptr), GOC_SUCCESS);
      hash = goc_test::capture_hash_word(hash, result);
      if ((index + 1) % (1U << 20) == 0) {
        unsigned block = (index + 1) / (1U << 20) - 1;
        ASSERT_EQ(hash, goc_test::unary_normalized_checkpoints[op][block])
            << "op=" << op << ", input indices [" << (block << 20) << ", " << (index + 1) << ')';
      }
    }
    EXPECT_EQ(hash, goc_test::unary_normalized_hashes[op]) << op;
  }
}

TEST(PseudoScalar, ExactHalfEveryEncodingAndGuestPolicy) {
  const goc_test::PseudoScalarFn functions[] = {goc_v_s_rcp_f16, goc_v_s_rsq_f16, goc_v_s_sqrt_f16};
  const unsigned reference_ops[] = {5, 6, 4};
  const uint64_t exact = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;
  const unsigned cpu_count = unsigned(goc_init_cpu_flags() + 1);
  for (unsigned op = 0; op < 3; ++op)
    for (unsigned state = 0; state < 8; ++state)
      for (unsigned raw = 0; raw < 65536; ++raw) {
        uint64_t flags = exact | goc_test::pseudo_scalar_flags(state) | (raw % cpu_count);
        uint32_t result;
        ASSERT_EQ(functions[op](flags, 0, &result, 0xabcd0000U | raw, nullptr), GOC_SUCCESS);
        ASSERT_EQ(result >> 16, 0U);
        if ((raw & 0x7fff) > 0x7c00) {
          ASSERT_EQ(result, raw | 0x200U);
          continue;
        }
        uint16_t input = uint16_t(raw);
        if ((flags & GOC_FP_FLUSH_INPUT_DENORMALS) && (input & 0x7fff) < 0x400)
          input &= 0x8000;
        uint16_t expected =
            goc_test::half_unary_reference(reference_ops[op], input, 0, flags & GOC_FP16_OVFL);
        if ((flags & GOC_FP_FLUSH_OUTPUT_DENORMALS) && (expected & 0x7fff) < 0x400)
          expected &= 0x8000;
        if ((expected & 0x7fff) > 0x7c00)
          ASSERT_EQ(result, 0xfe00U);
        else if ((expected & 0x7fff) && (expected & 0x7fff) < 0x7c00) {
          ASSERT_EQ(result & 0x8000, expected & 0x8000);
          ASSERT_TRUE(goc_test::fuzzy(goc_test::half_value(uint16_t(result)),
                                      goc_test::half_value(uint16_t(expected)),
                                      goc_test::half_sfu_tolerance));
        } else
          ASSERT_EQ(result, expected) << op << "/" << state << "/" << raw;
      }
}

// SPDX-License-Identifier: MIT

#include "fp_environment.h"
#include "goc/goc.h"
#include "rdna4_exec_masks.h"
#include "rdna4_pseudo_scalar_hardware.h"
#include "rdna4_pseudo_scalar_reference.h"

#include <cfenv>
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
                          cpu | goc_test::pseudo_scalar_flags(state), UINT32_MAX,
                          goc_test::pseudo_scalar_mode(m), &result, a),
                      GOC_SUCCESS);
            ASSERT_TRUE(goc_test::pseudo_scalar_close(result, gold[sample], op & 1))
                << state << "/" << op << "/" << m << "/" << sample << "/" << cpu << std::hex
                << " result " << result << " expected " << gold[sample];
          }
      }
}

TEST(PseudoScalar, ExecIsIgnoredAndScalarStorageMayAlias) {
  const uint64_t max_cpu = goc_init_cpu_flags();
  for (unsigned state = 0; state < 8; ++state)
    for (unsigned op = 0; op < 10; ++op)
      for (unsigned m = 0; m < 32; ++m) {
        const auto &gold =
            goc_test::pseudo_scalar_outputs[goc_test::pseudo_scalar_blocks[(state * 10 + op) * 32 +
                                                                           m]];
        for (unsigned sample : {0u, 1u, 2u, 9u, 16u, 20u, 42u})
          for (uint64_t cpu = 0; cpu <= max_cpu; ++cpu)
            for (uint32_t exec_mask : rdna4_exec_masks()) {
              uint32_t words[] = {
                  0x12345678,
                  goc_test::pseudo_scalar_input(op & 1, goc_test::pseudo_scalar_samples[sample]),
                  0x87654321};
              ASSERT_EQ(goc_test::pseudo_scalar_functions[op](
                            cpu | goc_test::pseudo_scalar_flags(state), exec_mask,
                            goc_test::pseudo_scalar_mode(m), words + 1, words[1]),
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
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    ASSERT_EQ(goc_rdna4_v_s_sqrt_f16(cpu, 0, 0, &d, 0xabcd0001), GOC_SUCCESS);
    EXPECT_EQ(d, 0x0c00u);
    ASSERT_EQ(goc_rdna4_v_s_sqrt_f16(cpu | GOC_FP_FLUSH_INPUT_DENORMALS, 0, 0, &d, 1), GOC_SUCCESS);
    EXPECT_EQ(d, 0u);
    ASSERT_EQ(goc_rdna4_v_s_rcp_f16(cpu, 0, 0, &d, 0x7bff), GOC_SUCCESS);
    EXPECT_EQ(d, 0x0100u);
    ASSERT_EQ(goc_rdna4_v_s_rcp_f16(cpu | GOC_FP_FLUSH_OUTPUT_DENORMALS, 0, 0, &d, 0x7bff),
              GOC_SUCCESS);
    EXPECT_EQ(d, 0u);
    ASSERT_EQ(goc_rdna4_v_s_rcp_f16(cpu, 0, GOC_ALU_OMOD_HALF, &d, 0xf00f), GOC_SUCCESS);
    EXPECT_EQ(d, 0x8000u);
    ASSERT_EQ(goc_rdna4_v_s_sqrt_f16(cpu, 0, GOC_ALU_OMOD_HALF, &d, 0x8000), GOC_SUCCESS);
    EXPECT_EQ(d, 0u);
    ASSERT_EQ(goc_rdna4_v_s_rcp_f16(cpu | GOC_FP16_OVFL, 0, 0, &d, 0x8000), GOC_SUCCESS);
    EXPECT_EQ(d, 0xfbffu);
    ASSERT_EQ(goc_rdna4_v_s_exp_f16(cpu | GOC_FP16_OVFL, 0, 0, &d, 0x7c00), GOC_SUCCESS);
    EXPECT_EQ(d, 0x7c00u);
    ASSERT_EQ(goc_rdna4_v_s_rcp_f32(cpu, 0, 0, &d, 1), GOC_SUCCESS);
    EXPECT_EQ(d, 0x7f800000u);
    ASSERT_EQ(goc_rdna4_v_s_rcp_f32(cpu, 0, 0, &d, 0x7f7fffff), GOC_SUCCESS);
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
        EXPECT_EQ(goc_test::pseudo_scalar_functions[op](0, 0, 1u << bit, &d, 0),
                  GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(d, 0xdeadbeef);
      }
    EXPECT_EQ(goc_test::pseudo_scalar_functions[op](
                  GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, known, &d, 0),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    EXPECT_EQ(d, 0xdeadbeef);
    EXPECT_EQ(goc_test::pseudo_scalar_functions[op](1ULL << 63, 0, known, &d, 0),
              GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(d, 0xdeadbeef);
    EXPECT_EQ(goc_test::pseudo_scalar_functions[op](GOC_SEMANTICS_EXACT_EMPIRICAL, 0, known, &d,
                                                    goc_test::pseudo_scalar_input(op & 1, 8)),
              GOC_SUCCESS);
    EXPECT_EQ(std::fegetround(), FE_TONEAREST);
  }
}

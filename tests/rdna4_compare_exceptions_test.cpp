// SPDX-License-Identifier: MIT

#include "fp_environment.h"
#include "goc/goc.h"
#include "rdna4_compare_exceptions_hardware.h"
#include "rdna4_dpp_reference.h"
#include "rdna4_float_compare_reference.h"

#include <cfenv>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdint.h>

namespace {

using ScalarCompare = decltype(&goc_rdna4_s_cmp_lt_f16);
const ScalarCompare scalar_functions[] = {
    goc_rdna4_s_cmp_lt_f16,  goc_rdna4_s_cmp_eq_f16,  goc_rdna4_s_cmp_le_f16,
    goc_rdna4_s_cmp_gt_f16,  goc_rdna4_s_cmp_lg_f16,  goc_rdna4_s_cmp_ge_f16,
    goc_rdna4_s_cmp_o_f16,   goc_rdna4_s_cmp_u_f16,   goc_rdna4_s_cmp_nge_f16,
    goc_rdna4_s_cmp_nlg_f16, goc_rdna4_s_cmp_ngt_f16, goc_rdna4_s_cmp_nle_f16,
    goc_rdna4_s_cmp_neq_f16, goc_rdna4_s_cmp_nlt_f16, goc_rdna4_s_cmp_lt_f32,
    goc_rdna4_s_cmp_eq_f32,  goc_rdna4_s_cmp_le_f32,  goc_rdna4_s_cmp_gt_f32,
    goc_rdna4_s_cmp_lg_f32,  goc_rdna4_s_cmp_ge_f32,  goc_rdna4_s_cmp_o_f32,
    goc_rdna4_s_cmp_u_f32,   goc_rdna4_s_cmp_nge_f32, goc_rdna4_s_cmp_nlg_f32,
    goc_rdna4_s_cmp_ngt_f32, goc_rdna4_s_cmp_nle_f32, goc_rdna4_s_cmp_neq_f32,
    goc_rdna4_s_cmp_nlt_f32};

} // namespace

TEST(CompareExceptions, HardwareOperandPairs) {
  for (unsigned signaling = 0; signaling < 2; ++signaling)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned denorm = 0; denorm < 4; ++denorm)
        for (uint32_t exec_mask : {UINT32_MAX, 0U, 1U}) {
          uint64_t flags = cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT |
                           (denorm & 1 ? 0 : GOC_FP_FLUSH_INPUT_DENORMALS) |
                           (denorm & 2 ? 0 : GOC_FP_FLUSH_OUTPUT_DENORMALS);
          for (unsigned op = signaling ? 28 : 0; op < 112; ++op) {
            unsigned format = op < 28 ? op / 14 : (op - 28) / 28;
            for (unsigned pair = 0; pair < 144; ++pair) {
              uint64_t av = goc_test::compare_exception_inputs[format][pair / 12];
              uint64_t bv = goc_test::compare_exception_inputs[format][pair % 12];
              uint32_t words[4][32];
              for (unsigned lane = 0; lane < 32; ++lane) {
                words[0][lane] = uint32_t(av);
                words[1][lane] = uint32_t(av >> 32);
                words[2][lane] = uint32_t(bv);
                words[3][lane] = uint32_t(bv >> 32);
              }
              const uint32_t *a[] = {words[0], words[1]}, *b[] = {words[2], words[3]};
              uint32_t result = 0, no_report_result = 0, exceptions = 0xa5000040;
              int status, no_report_status;
              if (op < 28) {
                status = scalar_functions[op](flags, 0, &result, uint32_t(av), uint32_t(bv),
                                              &exceptions);
                no_report_status = scalar_functions[op](flags, 0, &no_report_result, uint32_t(av),
                                                        uint32_t(bv), nullptr);
              } else {
                auto fn = goc_test::float_compare_functions[op - 28];
                status =
                    fn(flags, exec_mask, signaling ? GOC_ALU_CLAMP : 0, &result, a, b, &exceptions);
                no_report_status = fn(flags, exec_mask, signaling ? GOC_ALU_CLAMP : 0,
                                      &no_report_result, a, b, nullptr);
              }
              ASSERT_EQ(status, GOC_SUCCESS) << op;
              ASSERT_EQ(no_report_status, GOC_SUCCESS);
              EXPECT_EQ(result, no_report_result);
              uint32_t generated =
                  op < 28 || exec_mask
                      ? goc_test::compare_exception_hardware[(denorm & 1) + 2 * signaling][pair]
                      : 0;
              ASSERT_EQ(exceptions, 0xa5000040U | generated) << op << "/" << pair << "/" << denorm;
            }
          }
        }
}

TEST(CompareExceptions, LaneReductionDppAndHalfSelection) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned op = 0; op < 84; ++op)
      for (uint64_t descriptor : std::initializer_list<uint64_t>{
               0, GOC_DPP8 | (0x1f58d1ULL << 40),
               GOC_DPP16 | (0x101ULL << 40) | GOC_DPP_ROW_MASK | GOC_DPP_BANK_MASK}) {
        unsigned format = op / 28;
        if (format == 2 && descriptor)
          continue;
        for (uint32_t exec_mask : {UINT32_MAX, 0U, 0x55555555U, 0x80000000U}) {
          uint32_t words[4][32];
          for (unsigned lane = 0; lane < 32; ++lane) {
            uint64_t av = goc_test::compare_exception_inputs[format][lane % 12];
            uint64_t bv = goc_test::compare_exception_inputs[format][(lane * 5) % 12];
            words[0][lane] = uint32_t(av);
            words[1][lane] = uint32_t(av >> 32);
            words[2][lane] = uint32_t(bv);
            words[3][lane] = uint32_t(bv >> 32);
            if (format == 0) {
              words[0][lane] <<= 16;
              words[2][lane] <<= 16;
            }
          }
          uint32_t want = 0;
          for (unsigned lane = 0; lane < 32; ++lane) {
            int source = int(lane);
            bool active = descriptor ? goc_test::dpp_source(descriptor, exec_mask, lane, source)
                                     : bool((exec_mask >> lane) & 1);
            if (active) {
              unsigned a_index = source < 0 ? 0 : unsigned(source) % 12;
              want |= goc_test::compare_exception_hardware[3][a_index * 12 + (lane * 5) % 12];
            }
          }
          const uint32_t *a[] = {words[0], words[1]}, *b[] = {words[2], words[3]};
          uint64_t mode = descriptor | GOC_ALU_CLAMP | GOC_ALU_NEG_A | GOC_ALU_ABS_B |
                          (format == 0 ? GOC_ALU_HIGH_A | GOC_ALU_HIGH_B : 0);
          uint32_t exceptions = 0x80000000;
          // Destination aliases input: exception classification must precede stores.
          ASSERT_EQ(goc_test::float_compare_functions[op](cpu | GOC_SEMANTICS_EXACT_EMPIRICAL,
                                                          exec_mask, mode, words[0] + 1, a, b,
                                                          &exceptions),
                    GOC_SUCCESS);
          EXPECT_EQ(exceptions, 0x80000000U | want) << op;
        }
      }
}

TEST(CompareExceptions, LooseSkipsReportingAndErrorsPreserveOutputs) {
  uint32_t a[32] = {0x7f800001}, b[32] = {}, result = 0xdeadbeef;
  const uint32_t *ap[] = {a}, *bp[] = {b};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint32_t exceptions = 0x80000040;
    EXPECT_EQ(goc_rdna4_v_cmp_eq_f32(cpu, 1, 0, &result, ap, bp, &exceptions), GOC_SUCCESS);
    EXPECT_EQ(exceptions, 0x80000040U);
    result = 0xdeadbeef;
    EXPECT_EQ(goc_rdna4_v_cmp_eq_f32(cpu | GOC_SEMANTICS_EXACT_EMPIRICAL, 1, 1ULL << 63, &result,
                                     ap, bp, &exceptions),
              GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(result, 0xdeadbeefU);
    EXPECT_EQ(exceptions, 0x80000040U);
  }
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
    std::fesetround(rounding);
    std::feraiseexcept(FE_INEXACT);
    int host_exceptions = std::fetestexcept(FE_ALL_EXCEPT);
    uint32_t exceptions = 0;
    ASSERT_EQ(
        goc_rdna4_v_cmp_eq_f32(GOC_SEMANTICS_EXACT_EMPIRICAL, 1, 0, &result, ap, bp, &exceptions),
        GOC_SUCCESS);
    EXPECT_EQ(exceptions, GOC_RDNA4_EXCEPTION_INVALID);
    EXPECT_EQ(std::fegetround(), rounding);
    EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), host_exceptions);
  }
}

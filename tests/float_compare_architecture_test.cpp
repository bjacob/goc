// SPDX-License-Identifier: MIT

#include "dpp_reference.h"
#include "float_compare_reference.h"
#include "fp_environment.h"
#include "goc/goc.h"

#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdint.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace {

const goc_test::FloatCompareFn rdna3[] = {
    goc_v_cmp_lt_f16,   goc_v_cmp_eq_f16,   goc_v_cmp_le_f16,   goc_v_cmp_gt_f16,
    goc_v_cmp_lg_f16,   goc_v_cmp_ge_f16,   goc_v_cmp_o_f16,    goc_v_cmp_u_f16,
    goc_v_cmp_nge_f16,  goc_v_cmp_nlg_f16,  goc_v_cmp_ngt_f16,  goc_v_cmp_nle_f16,
    goc_v_cmp_neq_f16,  goc_v_cmp_nlt_f16,  goc_v_cmpx_lt_f16,  goc_v_cmpx_eq_f16,
    goc_v_cmpx_le_f16,  goc_v_cmpx_gt_f16,  goc_v_cmpx_lg_f16,  goc_v_cmpx_ge_f16,
    goc_v_cmpx_o_f16,   goc_v_cmpx_u_f16,   goc_v_cmpx_nge_f16, goc_v_cmpx_nlg_f16,
    goc_v_cmpx_ngt_f16, goc_v_cmpx_nle_f16, goc_v_cmpx_neq_f16, goc_v_cmpx_nlt_f16,
    goc_v_cmp_lt_f32,   goc_v_cmp_eq_f32,   goc_v_cmp_le_f32,   goc_v_cmp_gt_f32,
    goc_v_cmp_lg_f32,   goc_v_cmp_ge_f32,   goc_v_cmp_o_f32,    goc_v_cmp_u_f32,
    goc_v_cmp_nge_f32,  goc_v_cmp_nlg_f32,  goc_v_cmp_ngt_f32,  goc_v_cmp_nle_f32,
    goc_v_cmp_neq_f32,  goc_v_cmp_nlt_f32,  goc_v_cmpx_lt_f32,  goc_v_cmpx_eq_f32,
    goc_v_cmpx_le_f32,  goc_v_cmpx_gt_f32,  goc_v_cmpx_lg_f32,  goc_v_cmpx_ge_f32,
    goc_v_cmpx_o_f32,   goc_v_cmpx_u_f32,   goc_v_cmpx_nge_f32, goc_v_cmpx_nlg_f32,
    goc_v_cmpx_ngt_f32, goc_v_cmpx_nle_f32, goc_v_cmpx_neq_f32, goc_v_cmpx_nlt_f32,
    goc_v_cmp_lt_f64,   goc_v_cmp_eq_f64,   goc_v_cmp_le_f64,   goc_v_cmp_gt_f64,
    goc_v_cmp_lg_f64,   goc_v_cmp_ge_f64,   goc_v_cmp_o_f64,    goc_v_cmp_u_f64,
    goc_v_cmp_nge_f64,  goc_v_cmp_nlg_f64,  goc_v_cmp_ngt_f64,  goc_v_cmp_nle_f64,
    goc_v_cmp_neq_f64,  goc_v_cmp_nlt_f64,  goc_v_cmpx_lt_f64,  goc_v_cmpx_eq_f64,
    goc_v_cmpx_le_f64,  goc_v_cmpx_gt_f64,  goc_v_cmpx_lg_f64,  goc_v_cmpx_ge_f64,
    goc_v_cmpx_o_f64,   goc_v_cmpx_u_f64,   goc_v_cmpx_nge_f64, goc_v_cmpx_nlg_f64,
    goc_v_cmpx_ngt_f64, goc_v_cmpx_nle_f64, goc_v_cmpx_neq_f64, goc_v_cmpx_nlt_f64};
const decltype(&goc_v_cmp_lt_f16_wave64) rdna3_wave64[] = {
    goc_v_cmp_lt_f16_wave64,   goc_v_cmp_eq_f16_wave64,   goc_v_cmp_le_f16_wave64,
    goc_v_cmp_gt_f16_wave64,   goc_v_cmp_lg_f16_wave64,   goc_v_cmp_ge_f16_wave64,
    goc_v_cmp_o_f16_wave64,    goc_v_cmp_u_f16_wave64,    goc_v_cmp_nge_f16_wave64,
    goc_v_cmp_nlg_f16_wave64,  goc_v_cmp_ngt_f16_wave64,  goc_v_cmp_nle_f16_wave64,
    goc_v_cmp_neq_f16_wave64,  goc_v_cmp_nlt_f16_wave64,  goc_v_cmpx_lt_f16_wave64,
    goc_v_cmpx_eq_f16_wave64,  goc_v_cmpx_le_f16_wave64,  goc_v_cmpx_gt_f16_wave64,
    goc_v_cmpx_lg_f16_wave64,  goc_v_cmpx_ge_f16_wave64,  goc_v_cmpx_o_f16_wave64,
    goc_v_cmpx_u_f16_wave64,   goc_v_cmpx_nge_f16_wave64, goc_v_cmpx_nlg_f16_wave64,
    goc_v_cmpx_ngt_f16_wave64, goc_v_cmpx_nle_f16_wave64, goc_v_cmpx_neq_f16_wave64,
    goc_v_cmpx_nlt_f16_wave64, goc_v_cmp_lt_f32_wave64,   goc_v_cmp_eq_f32_wave64,
    goc_v_cmp_le_f32_wave64,   goc_v_cmp_gt_f32_wave64,   goc_v_cmp_lg_f32_wave64,
    goc_v_cmp_ge_f32_wave64,   goc_v_cmp_o_f32_wave64,    goc_v_cmp_u_f32_wave64,
    goc_v_cmp_nge_f32_wave64,  goc_v_cmp_nlg_f32_wave64,  goc_v_cmp_ngt_f32_wave64,
    goc_v_cmp_nle_f32_wave64,  goc_v_cmp_neq_f32_wave64,  goc_v_cmp_nlt_f32_wave64,
    goc_v_cmpx_lt_f32_wave64,  goc_v_cmpx_eq_f32_wave64,  goc_v_cmpx_le_f32_wave64,
    goc_v_cmpx_gt_f32_wave64,  goc_v_cmpx_lg_f32_wave64,  goc_v_cmpx_ge_f32_wave64,
    goc_v_cmpx_o_f32_wave64,   goc_v_cmpx_u_f32_wave64,   goc_v_cmpx_nge_f32_wave64,
    goc_v_cmpx_nlg_f32_wave64, goc_v_cmpx_ngt_f32_wave64, goc_v_cmpx_nle_f32_wave64,
    goc_v_cmpx_neq_f32_wave64, goc_v_cmpx_nlt_f32_wave64, goc_v_cmp_lt_f64_wave64,
    goc_v_cmp_eq_f64_wave64,   goc_v_cmp_le_f64_wave64,   goc_v_cmp_gt_f64_wave64,
    goc_v_cmp_lg_f64_wave64,   goc_v_cmp_ge_f64_wave64,   goc_v_cmp_o_f64_wave64,
    goc_v_cmp_u_f64_wave64,    goc_v_cmp_nge_f64_wave64,  goc_v_cmp_nlg_f64_wave64,
    goc_v_cmp_ngt_f64_wave64,  goc_v_cmp_nle_f64_wave64,  goc_v_cmp_neq_f64_wave64,
    goc_v_cmp_nlt_f64_wave64,  goc_v_cmpx_lt_f64_wave64,  goc_v_cmpx_eq_f64_wave64,
    goc_v_cmpx_le_f64_wave64,  goc_v_cmpx_gt_f64_wave64,  goc_v_cmpx_lg_f64_wave64,
    goc_v_cmpx_ge_f64_wave64,  goc_v_cmpx_o_f64_wave64,   goc_v_cmpx_u_f64_wave64,
    goc_v_cmpx_nge_f64_wave64, goc_v_cmpx_nlg_f64_wave64, goc_v_cmpx_ngt_f64_wave64,
    goc_v_cmpx_nle_f64_wave64, goc_v_cmpx_neq_f64_wave64, goc_v_cmpx_nlt_f64_wave64};

uint64_t policy(unsigned guest) {
  return (guest & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
         (guest & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) | (guest & 4 ? GOC_FP_IEEE_MODE : 0);
}

} // namespace

TEST(FloatCompareArchitecture, GuestPoliciesModifiersAndNativeState) {
  for (unsigned op = 0; op < 84; ++op) {
    SCOPED_TRACE(goc_test::float_compare_names[op]);
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned guest = 0; guest < 8; ++guest)
        for (unsigned m = 0; m < (op < 28 ? 64u : 16u); ++m)
          for (unsigned start = 0; start < 320; start += 64) {
            uint32_t words[4][64];
            uint64_t expected = 0;
            const uint64_t exec_mask = start == 256 ? 0x9249249281818181ULL : UINT64_MAX;
            for (unsigned lane = 0; lane < 64; ++lane) {
              uint32_t w[4];
              goc_test::float_compare_inputs(op / 28, start + lane, w);
              for (unsigned j = 0; j < 4; ++j)
                words[j][lane] = w[j];
              expected |= uint64_t(goc_test::float_compare_reference(op, m, guest & 1, w)) << lane;
            }
            expected &= exec_mask;
            const uint32_t *a[] = {words[0], words[1]}, *b[] = {words[2], words[3]};
            const uint64_t mode = goc_test::float_compare_mode(m) | GOC_ALU_CLAMP;
            for (uint64_t semantics :
                 {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT}) {
              const uint64_t flags = cpu | policy(guest) | semantics;
              uint32_t state = 0x12345678, result32 = 0;
              uint64_t result64 = 0;
              uint32_t *state_output = semantics == GOC_SEMANTICS_LOOSE ? &state : nullptr;
              ASSERT_EQ(rdna3[op](flags, uint32_t(exec_mask), mode, &result32, a, b, state_output),
                        GOC_SUCCESS);
              ASSERT_EQ(result32, uint32_t(expected));
              ASSERT_EQ(rdna3_wave64[op](flags, exec_mask, mode, &result64, a, b, state_output),
                        GOC_SUCCESS);
              ASSERT_EQ(result64, expected);
              if (state_output) {
                ASSERT_EQ(state, 0x12345678U);
              }
              ASSERT_EQ(goc_test::float_compare_functions[op](flags, uint32_t(exec_mask), mode,
                                                              &result32, a, b, state_output),
                        GOC_SUCCESS);
              ASSERT_EQ(result32, uint32_t(expected));
              ASSERT_EQ(goc_test::float_compare_wave64_functions[op](flags, exec_mask, mode,
                                                                     &result64, a, b, state_output),
                        GOC_SUCCESS);
              ASSERT_EQ(result64, expected);
              if (state_output) {
                ASSERT_EQ(state, 0x12345678U);
              }
            }
          }
  }
}

TEST(FloatCompareArchitecture, UnsupportedReportingIsAtomicIncludingEmptyExec) {
  for (unsigned op = 0; op < 84; ++op)
    for (uint64_t strict : {uint64_t(0), GOC_SEMANTICS_STRICT})
      for (uint64_t exec_mask : {uint64_t(0), UINT64_MAX}) {
        uint32_t state = 0x13579bdf, result32 = 0xabcdef01;
        uint64_t result64 = 0x123456789abcdef0ULL;
        const uint64_t flags = GOC_SEMANTICS_EXACT_EMPIRICAL | strict;
        EXPECT_EQ(rdna3[op](flags, uint32_t(exec_mask), 0, &result32, nullptr, nullptr, &state),
                  GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
        EXPECT_EQ(rdna3_wave64[op](flags, exec_mask, 0, &result64, nullptr, nullptr, &state),
                  GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
        EXPECT_EQ(state, 0x13579bdfU);
        EXPECT_EQ(result32, 0xabcdef01U);
        EXPECT_EQ(result64, 0x123456789abcdef0ULL);
        // The ordinary predicate output is still required when reporting is disabled.
        EXPECT_EQ(rdna3[op](flags, 0, 0, &result32, nullptr, nullptr, nullptr), GOC_SUCCESS);
        EXPECT_EQ(rdna3_wave64[op](flags, 0, 0, &result64, nullptr, nullptr, nullptr), GOC_SUCCESS);
        EXPECT_EQ(result32, 0U);
        EXPECT_EQ(result64, 0ULL);
      }
}

TEST(FloatCompareArchitecture, DppAndAliasedPredicateOutputs) {
  for (unsigned op = 0; op < 84; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned guest = 0; guest < 8; ++guest)
        for (uint64_t dpp : goc_test::dpp_modes) {
          if (op >= 56)
            continue;
          const uint64_t exec_mask = 0xa5a55a5af0f00f0fULL;
          uint32_t words[4][64];
          for (unsigned lane = 0; lane < 64; ++lane) {
            uint32_t w[4];
            goc_test::float_compare_inputs(op / 28, lane * 5, w);
            for (unsigned j = 0; j < 4; ++j)
              words[j][lane] = w[j];
          }
          uint64_t expected = 0;
          const unsigned m = op < 28 ? 37 : 5;
          for (unsigned lane = 0; lane < 64; ++lane) {
            int source;
            if (!goc_test::dpp_source(dpp, exec_mask, lane, source))
              continue;
            uint32_t w[] = {source < 0 ? 0 : words[0][source], 0, words[2][lane], 0};
            expected |= uint64_t(goc_test::float_compare_reference(op, m, guest & 1, w)) << lane;
          }
          const uint32_t *a[] = {words[0], words[1]}, *b[] = {words[2], words[3]};
          const uint64_t flags =
              cpu | policy(guest) | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;
          const uint64_t mode = dpp | goc_test::float_compare_mode(m);
          uint32_t saved[2];
          std::memcpy(saved, &words[0][33], sizeof(saved));
          auto *aliased = reinterpret_cast<uint64_t *>(&words[0][33]);
          ASSERT_EQ(rdna3_wave64[op](flags, exec_mask, mode, aliased, a, b, nullptr), GOC_SUCCESS);
          uint64_t result;
          std::memcpy(&result, aliased, sizeof(result));
          ASSERT_EQ(result, expected);
          std::memcpy(&words[0][33], saved, sizeof(saved));
          ASSERT_EQ(rdna3[op](flags, uint32_t(exec_mask), mode, &words[0][1], a, b, nullptr),
                    GOC_SUCCESS);
          ASSERT_EQ(words[0][1], uint32_t(expected));
        }
}

TEST(FloatCompareArchitecture, GuestPoliciesPreserveHostEnvironment) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO})
    for (bool host_flush : {false, true}) {
      ASSERT_EQ(std::fesetround(rounding), 0);
      std::feclearexcept(FE_ALL_EXCEPT);
      std::feraiseexcept(FE_DIVBYZERO);
#if defined(__x86_64__) || defined(_M_X64)
      _mm_setcsr((_mm_getcsr() & ~0x8040U) | (host_flush ? 0x8040U : 0));
      const unsigned mxcsr = _mm_getcsr();
#endif
      const int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
      for (unsigned op = 0; op < 84; ++op) {
        uint32_t words[4][64];
        for (unsigned lane = 0; lane < 64; ++lane) {
          uint32_t w[4];
          goc_test::float_compare_inputs(op / 28, lane * 5, w);
          for (unsigned j = 0; j < 4; ++j)
            words[j][lane] = w[j];
        }
        const uint32_t *a[] = {words[0], words[1]}, *b[] = {words[2], words[3]};
        for (unsigned guest = 0; guest < 8; ++guest)
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
            uint32_t result32;
            uint64_t result64;
            const uint64_t flags =
                cpu | policy(guest) | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;
            ASSERT_EQ(rdna3[op](flags, UINT32_MAX, 0, &result32, a, b, nullptr), GOC_SUCCESS);
            ASSERT_EQ(rdna3_wave64[op](flags, UINT64_MAX, 0, &result64, a, b, nullptr),
                      GOC_SUCCESS);
            uint32_t state = 0;
            ASSERT_EQ(goc_test::float_compare_functions[op](flags, UINT32_MAX, 0, &result32, a, b,
                                                            &state),
                      GOC_SUCCESS);
            ASSERT_EQ(goc_test::float_compare_wave64_functions[op](flags, UINT64_MAX, 0, &result64,
                                                                   a, b, &state),
                      GOC_SUCCESS);
          }
      }
      EXPECT_EQ(std::fegetround(), rounding);
      EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
#if defined(__x86_64__) || defined(_M_X64)
      EXPECT_EQ(_mm_getcsr(), mxcsr);
#endif
    }
}

// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "dpp_reference.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "unary_capture_input.h"
#include "unary_exact_hardware.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>

namespace {

uint32_t modifiers(unsigned m) {
  return (m & 1 ? GOC_ALU_NEG_A : 0) | (m & 2 ? GOC_ALU_ABS_A : 0) | ((m >> 2 & 3) << 6) |
         (m & 16 ? GOC_ALU_CLAMP : 0);
}

int invoke(unsigned op, bool rdna3, bool wide, uint64_t flags, uint64_t exec_mask, uint64_t mode,
           uint32_t *d, const uint32_t *a, uint32_t *state = nullptr) {
  const decltype(&goc_v_rcp_f32) narrow[2][5] = {
      {goc_v_rcp_f32_rdna4, goc_v_sqrt_f32_rdna4, goc_v_rsq_f32_rdna4, goc_v_exp_f32_rdna4,
       goc_v_log_f32_rdna4},
      {goc_v_rcp_f32, goc_v_sqrt_f32, goc_v_rsq_f32, goc_v_exp_f32, goc_v_log_f32}};
  const decltype(&goc_v_rcp_f32_wave64) wave64[2][5] = {
      {goc_v_rcp_f32_rdna4_wave64, goc_v_sqrt_f32_rdna4_wave64, goc_v_rsq_f32_rdna4_wave64,
       goc_v_exp_f32_rdna4_wave64, goc_v_log_f32_rdna4_wave64},
      {goc_v_rcp_f32_wave64, goc_v_sqrt_f32_wave64, goc_v_rsq_f32_wave64, goc_v_exp_f32_wave64,
       goc_v_log_f32_wave64}};
  return wide ? wave64[rdna3][op](flags, exec_mask, mode, &d, &a, state)
              : narrow[rdna3][op](flags, uint32_t(exec_mask), mode, &d, &a, state);
}

class UnaryExact : public ::testing::TestWithParam<unsigned> {};

} // namespace

TEST_P(UnaryExact, DenseDomainHardwareCapture) {
  const unsigned op = GetParam();
  uint64_t hash = goc_test::capture_hash_seed;
  for (unsigned base = 0; base < (1U << (op == 0 ? 23 : op == 3 ? 25 : 24)); base += 64) {
    uint32_t input[64], output[64];
    for (unsigned lane = 0; lane < 64; ++lane) {
      unsigned index = base + lane;
      if (op == 3)
        input[lane] = (0x3f000000 + (index & 0xffffff)) | ((index >> 24) << 31);
      else if (op == 4)
        input[lane] = 0x3f000000 + index;
      else
        input[lane] = 0x3f800000 + index;
    }
    bool wide = (base >> 6) & 1, rdna3 = (base >> 7) & 1;
    uint64_t flags = ((base >> 8) % (goc_init_cpu_flags() + 1)) | GOC_SEMANTICS_EXACT_EMPIRICAL |
                     GOC_SEMANTICS_STRICT;
    ASSERT_EQ(invoke(op, rdna3, wide, flags, UINT64_MAX, 0, output, input), GOC_SUCCESS);
    if (!wide) {
      ASSERT_EQ(invoke(op, rdna3, false, flags, UINT32_MAX, 0, output + 32, input + 32),
                GOC_SUCCESS);
    }
    for (auto value : output)
      hash = goc_test::capture_hash_word(hash, value);
  }
  EXPECT_EQ(hash, goc_test::unary_normalized_hashes[op]);
}

TEST_P(UnaryExact, ModifiersExponentsSpecialValuesHardwareCapture) {
  const unsigned op = GetParam();
  for (bool wide : {false, true})
    for (unsigned policy = 0; policy < 4; ++policy)
      for (unsigned m = 0; m < 32; ++m) {
        uint64_t hash = goc_test::capture_hash_seed;
        unsigned lanes = wide ? 64 : 32;
        for (unsigned base = 0; base < 8192; base += lanes) {
          uint32_t input[64], output[64];
          for (unsigned lane = 0; lane < lanes; ++lane)
            input[lane] = goc_test::unary_capture_input(op, base + lane);
          uint64_t flags = ((base / lanes) % (goc_init_cpu_flags() + 1)) |
                           GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT |
                           (policy & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                           (policy & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0);
          ASSERT_EQ(invoke(op, false, wide, flags, UINT64_MAX, modifiers(m), output, input),
                    GOC_SUCCESS);
          for (unsigned lane = 0; lane < lanes; ++lane)
            hash = goc_test::capture_hash_word(hash, output[lane]);
        }
        EXPECT_EQ(hash, goc_test::unary_modifier_hashes[op][policy][m])
            << wide << '/' << policy << '/' << m;
      }
}

TEST_P(UnaryExact, ArchitectureNaNPriorityAndOmod) {
  const unsigned op = GetParam();
  for (bool rdna3 : {false, true})
    for (bool ieee : {false, true})
      for (bool wide : {false, true}) {
        uint32_t input[64], output[64];
        for (unsigned lane = 0; lane < 64; ++lane)
          input[lane] = lane & 1 ? 0xff800123 : 0x40800000;
        uint64_t flags =
            GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT | (ieee ? GOC_FP_IEEE_MODE : 0);
        ASSERT_EQ(invoke(op, rdna3, wide, flags, UINT64_MAX, GOC_ALU_OMOD_2, output, input),
                  GOC_SUCCESS);
        const uint32_t unscaled[] = {0x3e800000, 0x40000000, 0x3f000000, 0x41800000, 0x40000000};
        const uint32_t scaled[] = {0x3f000000, 0x40800000, 0x3f800000, 0x42000000, 0x40800000};
        uint32_t finite = rdna3 && ieee ? unscaled[op] : scaled[op];
        uint32_t nan = rdna3 && !ieee ? 0xff800123 : 0xffc00123;
        for (unsigned lane = 0; lane < (wide ? 64U : 32U); ++lane)
          EXPECT_EQ(output[lane], lane & 1 ? nan : finite);
      }
}

TEST_P(UnaryExact, DppAliasesMasksAndHostEnvironment) {
  const unsigned op = GetParam();
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (bool rdna3 : {false, true})
    for (bool wide : {false, true})
      for (bool alias : {false, true})
        for (unsigned m = 0; m < 32; ++m)
          for (auto descriptor : goc_test::dpp_modes) {
            uint32_t original[64], expected[64];
            for (unsigned lane = 0; lane < 64; ++lane)
              original[lane] = goc_test::unary_capture_input(op, lane % 32);
            unsigned lanes = wide ? 64 : 32;
            uint64_t exec_mask = wide ? 0x9192a5a571abcdefULL : 0x71abcdef;
            uint64_t flags = (m % (goc_init_cpu_flags() + 1)) | GOC_SEMANTICS_EXACT_EMPIRICAL |
                             GOC_SEMANTICS_STRICT | (m & 1 ? GOC_FP_IEEE_MODE : 0);
            uint32_t expected_state = 0x80000040;
            for (unsigned lane = 0; lane < lanes; ++lane) {
              expected[lane] = alias ? original[lane] : 0xdeadbeef;
              int source;
              if (goc_test::dpp_source(descriptor, exec_mask, lane, source)) {
                uint32_t input[32] = {}, output[32];
                input[0] = source < 0 ? 0 : original[source];
                ASSERT_EQ(invoke(op, rdna3, false, flags, 1, modifiers(m), output, input, nullptr),
                          GOC_SUCCESS);
                expected[lane] = output[0];
              }
            }
            for (int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO})
              for (bool flush : {false, true}) {
                ASSERT_EQ(std::fesetround(rounding), 0);
                std::feclearexcept(FE_ALL_EXCEPT);
                std::feraiseexcept(FE_DIVBYZERO);
#if defined(__x86_64__) || defined(_M_X64)
                _mm_setcsr((_mm_getcsr() & ~0x8040U) | (flush ? 0x8040U : 0));
                unsigned controls = _mm_getcsr();
#endif
                uint32_t input[64], output[64];
                std::memcpy(input, original, sizeof(input));
                std::fill(output, output + 64, 0xdeadbeef);
                uint32_t *d = alias ? input : output;
                uint32_t state = 0x80000040;
                ASSERT_EQ(invoke(op, rdna3, wide, flags, exec_mask, modifiers(m) | descriptor, d,
                                 input, nullptr),
                          GOC_SUCCESS);
                EXPECT_EQ(std::memcmp(d, expected, lanes * sizeof(uint32_t)), 0);
                if (!rdna3) {
                  EXPECT_EQ(state, expected_state);
                }
                EXPECT_EQ(std::fegetround(), rounding);
                EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
#if defined(__x86_64__) || defined(_M_X64)
                EXPECT_EQ(_mm_getcsr(), controls);
#endif
              }
          }
}

TEST_P(UnaryExact, ReportingRemainsExplicitlyUnsupported) {
  const unsigned op = GetParam();
  for (bool rdna3 : {false, true})
    for (bool wide : {false, true}) {
      uint32_t state = 0x80000001;
      for (uint64_t strict : {uint64_t(0), uint64_t(GOC_SEMANTICS_STRICT)}) {
        EXPECT_EQ(invoke(op, rdna3, wide, GOC_SEMANTICS_EXACT_EMPIRICAL | strict, UINT64_MAX, 0,
                         nullptr, nullptr, &state),
                  GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
        EXPECT_EQ(state, 0x80000001U);
      }
    }
}

INSTANTIATE_TEST_SUITE_P(Captured, UnaryExact, ::testing::Values(0U, 1U, 2U, 3U, 4U));

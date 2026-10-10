// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "dpp_reference.h"
#include "fp_environment.h"
#include "fract_hardware.h"
#include "goc/goc.h"
#include "unary_capture_input.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>

namespace {

uint32_t capture_input(unsigned corpus, unsigned i) {
  if (!corpus)
    return goc_test::unary_capture_input(0, i);
  const int offsets[] = {0, 1, -1, 0x400000};
  uint32_t word = (((i / 32) % 254 + 1) << 23) + offsets[i % 4];
  return word | (((i % 32) / 4 % 2) << 31);
}

int invoke(bool rdna3, bool wide, uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *d,
           const uint32_t *a, uint32_t *state) {
  if (wide)
    return (rdna3 ? goc_v_fract_f32_wave64 : goc_v_fract_f32_rdna4_wave64)(flags, exec_mask, mode,
                                                                           &d, &a, state);
  return (rdna3 ? goc_v_fract_f32 : goc_v_fract_f32_rdna4)(flags, uint32_t(exec_mask), mode, &d, &a,
                                                           state);
}

uint32_t modifiers(unsigned m) {
  return (m & 1 ? GOC_ALU_NEG_A : 0) | (m & 2 ? GOC_ALU_ABS_A : 0) | ((m >> 2 & 3) << 6) |
         (m & 16 ? GOC_ALU_CLAMP : 0);
}

} // namespace

TEST(Fract, HardwareNumericsAndArchitecturePolicy) {
  for (unsigned corpus = 0; corpus < 2; ++corpus)
    for (bool exact : {false, true})
      for (bool rdna3 : {false, true})
        for (bool wide : {false, true})
          for (unsigned policy = 0; policy < 8; ++policy)
            for (unsigned m = 0; m < 32; ++m) {
              unsigned lanes = wide ? 64 : 32;
              uint64_t hash = goc_test::capture_hash_seed;
              for (unsigned base = 0; base < 8192; base += lanes) {
                uint32_t input[64], output[64];
                for (unsigned lane = 0; lane < lanes; ++lane)
                  input[lane] = capture_input(corpus, base + lane);
                uint64_t flags =
                    ((base / lanes) % (goc_init_cpu_flags() + 1)) |
                    (exact ? GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT : 0) |
                    (policy & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                    (policy & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) |
                    (policy & 4 ? GOC_FP_IEEE_MODE : 0);
                ASSERT_EQ(
                    invoke(rdna3, wide, flags, UINT64_MAX, modifiers(m), output, input, nullptr),
                    GOC_SUCCESS);
                for (unsigned lane = 0; lane < lanes; ++lane) {
                  uint32_t word = output[lane];
                  if ((word & 0x7fffffff) > 0x7f800000)
                    word = 0x7fc00000;
                  hash = goc_test::capture_hash_word(hash, word);
                }
              }
              unsigned effective = rdna3 && ((policy & 4) || !(policy & 2)) ? m & ~12U : m;
              EXPECT_EQ(hash, goc_test::fract_hardware[corpus][policy & 3][effective][0])
                  << exact << '/' << rdna3 << '/' << wide << '/' << policy << '/' << m;
            }
}

TEST(Fract, ExactReportingHardware) {
  for (unsigned corpus = 0; corpus < 2; ++corpus)
    for (unsigned policy = 0; policy < 4; ++policy)
      for (unsigned m = 0; m < 32; ++m) {
        uint64_t hash = goc_test::capture_hash_seed;
        for (unsigned i = 0; i < 8192; ++i) {
          bool wide = i & 1;
          unsigned lane = (i >> 1) % (wide ? 64 : 32);
          uint32_t input[64], output[64], state = 0x80000040;
          std::fill(input, input + 64, i & 1 ? 0x7f800001 : 1);
          std::fill(output, output + 64, 0xdeadbeef);
          input[lane] = capture_input(corpus, i);
          uint64_t flags =
              GOC_SEMANTICS_EXACT_EMPIRICAL | (policy & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
              (i & 2 ? GOC_SEMANTICS_STRICT : 0) |
              (policy & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) | (i & 8 ? GOC_FP_IEEE_MODE : 0);
          ASSERT_EQ(invoke(false, wide, flags, 1ULL << lane, modifiers(m), output, input, &state),
                    GOC_SUCCESS);
          ASSERT_EQ(state & 0x80000040U, 0x80000040U);
          EXPECT_EQ(output[(lane + 1) % (wide ? 64 : 32)], 0xdeadbeefU);
          hash = goc_test::capture_hash_word(hash, output[lane]);
          hash = goc_test::capture_hash_word(hash, state & ~0x80000040U);
        }
        EXPECT_EQ(hash, goc_test::fract_hardware[corpus][policy][m][1]);
      }
}

TEST(Fract, DppAliasesGuestModesAndHostControls) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (bool exact : {false, true})
    for (bool rdna3 : {false, true})
      for (bool wide : {false, true})
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
          for (unsigned policy = 0; policy < 8; ++policy)
            for (auto descriptor : goc_test::dpp_modes) {
              unsigned lanes = wide ? 64 : 32;
              uint64_t exec_mask = wide ? 0x9192a5a571abcdefULL : 0x71abcdef;
              uint64_t flags = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT |
                               (policy & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                               (policy & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) |
                               (policy & 4 ? GOC_FP_IEEE_MODE : 0);
              unsigned m = (policy * 5) % 32;
              unsigned effective = rdna3 && ((policy & 4) || !(policy & 2)) ? m & ~12U : m;
              uint32_t original[64], expected[64], expected_state = 0x80000040;
              for (unsigned lane = 0; lane < lanes; ++lane)
                original[lane] = goc_test::unary_capture_input(0, lane % 32);
              for (unsigned lane = 0; lane < lanes; ++lane) {
                expected[lane] = original[lane];
                int source;
                if (goc_test::dpp_source(descriptor, exec_mask, lane, source)) {
                  uint32_t input[32] = {}, output[32];
                  input[0] = source < 0 ? 0 : original[source];
                  ASSERT_EQ(invoke(false, false, flags, 1, modifiers(effective), output, input,
                                   &expected_state),
                            GOC_SUCCESS);
                  expected[lane] = output[0];
                }
              }
              flags = (exact ? flags : flags & ~(GOC_SEMANTICS_MASK | GOC_SEMANTICS_STRICT)) | cpu;
              for (int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO})
                for (unsigned host_policy = 0; host_policy < 4; ++host_policy) {
                  if (!exact && (rounding != FE_TONEAREST || host_policy))
                    continue;
                  ASSERT_EQ(std::fesetround(rounding), 0);
                  std::feclearexcept(FE_ALL_EXCEPT);
                  std::feraiseexcept(FE_DIVBYZERO);
#if defined(__x86_64__) || defined(_M_X64)
                  _mm_setcsr((_mm_getcsr() & ~0x8040U) | (host_policy & 1 ? 0x40U : 0) |
                             (host_policy & 2 ? 0x8000U : 0));
                  unsigned controls = _mm_getcsr();
#endif
                  uint32_t input[64], state = 0x80000040;
                  std::memcpy(input, original, lanes * sizeof(uint32_t));
                  ASSERT_EQ(invoke(rdna3, wide, flags, exec_mask, modifiers(m) | descriptor, input,
                                   input, rdna3 && exact ? nullptr : &state),
                            GOC_SUCCESS);
                  for (unsigned lane = 0; lane < lanes; ++lane) {
                    if (!exact && (expected[lane] & 0x7fffffff) > 0x7f800000) {
                      EXPECT_GT(input[lane] & 0x7fffffff, 0x7f800000U);
                    } else {
                      EXPECT_EQ(input[lane], expected[lane]);
                    }
                  }
                  EXPECT_EQ(state, exact && !rdna3 ? expected_state : 0x80000040U);
                  EXPECT_EQ(std::fegetround(), rounding);
                  EXPECT_EQ(std::fetestexcept(FE_DIVBYZERO), FE_DIVBYZERO);
#if defined(__x86_64__) || defined(_M_X64)
                  EXPECT_EQ(_mm_getcsr() & ~0x3fU, controls & ~0x3fU);
#endif
                }
            }
}

TEST(Fract, ValidationAndEmptyExec) {
  for (bool rdna3 : {false, true})
    for (bool wide : {false, true}) {
      uint32_t input[64] = {}, output[64], state = 0x80000040;
      std::fill(output, output + 64, 0xdeadbeef);
      EXPECT_EQ(invoke(rdna3, wide, 0, UINT64_MAX, GOC_ALU_NEG_B, output, input, &state),
                GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(invoke(rdna3, wide, 2ULL << 16, UINT64_MAX, 0, output, input, &state),
                GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
      EXPECT_EQ(invoke(rdna3, wide, GOC_SEMANTICS_EXACT_EMPIRICAL, 0, 0, nullptr, nullptr, &state),
                rdna3 ? GOC_ERROR_UNSUPPORTED_GLOBAL_STATE : GOC_SUCCESS);
      EXPECT_EQ(invoke(rdna3, wide, GOC_SEMANTICS_EXACT_EMPIRICAL, 0, 0, nullptr, nullptr, nullptr),
                GOC_SUCCESS);
      EXPECT_EQ(state, 0x80000040U);
      for (auto value : output)
        EXPECT_EQ(value, 0xdeadbeefU);
    }
}

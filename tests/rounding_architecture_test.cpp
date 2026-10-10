// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "dpp_reference.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "internal.h"
#include "rounding_hardware.h"
#include "unary_capture_input.h"
#include "unary_reference.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>

namespace {

int invoke(unsigned op, bool rdna3, bool wide, uint64_t flags, uint64_t exec_mask, uint64_t mode,
           uint32_t *d, const uint32_t *a, uint32_t *state) {
  const decltype(&goc_v_trunc_f32) narrow[2][4] = {
      {goc_v_trunc_f32_rdna4, goc_v_ceil_f32_rdna4, goc_v_rndne_f32_rdna4, goc_v_floor_f32_rdna4},
      {goc_v_trunc_f32, goc_v_ceil_f32, goc_v_rndne_f32, goc_v_floor_f32}};
  const decltype(&goc_v_trunc_f32_wave64) wave64[2][4] = {
      {goc_v_trunc_f32_rdna4_wave64, goc_v_ceil_f32_rdna4_wave64, goc_v_rndne_f32_rdna4_wave64,
       goc_v_floor_f32_rdna4_wave64},
      {goc_v_trunc_f32_wave64, goc_v_ceil_f32_wave64, goc_v_rndne_f32_wave64,
       goc_v_floor_f32_wave64}};
  return wide ? wave64[rdna3][op](flags, exec_mask, mode, &d, &a, state)
              : narrow[rdna3][op](flags, uint32_t(exec_mask), mode, &d, &a, state);
}

uint32_t modifiers(unsigned m) {
  return (m & 1 ? GOC_ALU_NEG_A : 0) | (m & 2 ? GOC_ALU_ABS_A : 0) | ((m >> 2 & 3) << 6) |
         (m & 16 ? GOC_ALU_CLAMP : 0);
}

} // namespace

TEST(RoundingArchitecture, Rdna4HardwareAndRdna3OmodPolicy) {
  for (unsigned op = 0; op < 4; ++op)
    for (bool rdna3 : {false, true})
      for (bool wide : {false, true})
        for (unsigned policy = 0; policy < 8; ++policy)
          for (unsigned m = 0; m < 32; ++m) {
            uint64_t hash = goc_test::capture_hash_seed;
            unsigned lanes = wide ? 64 : 32;
            for (unsigned base = 0; base < 8192; base += lanes) {
              uint32_t input[64], output[64];
              for (unsigned lane = 0; lane < lanes; ++lane)
                input[lane] = goc_test::unary_capture_input(0, base + lane);
              uint64_t flags = ((base / lanes) % (goc_init_cpu_flags() + 1)) |
                               (policy & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                               (policy & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) |
                               (policy & 4 ? GOC_FP_IEEE_MODE : 0);
              uint32_t state = 0x80000055;
              ASSERT_EQ(
                  invoke(op, rdna3, wide, flags, UINT64_MAX, modifiers(m), output, input, &state),
                  GOC_SUCCESS);
              ASSERT_EQ(state, 0x80000055U);
              for (unsigned lane = 0; lane < lanes; ++lane) {
                uint32_t word = output[lane];
                if ((word & 0x7fffffff) > 0x7f800000)
                  word = 0x7fc00000;
                hash = goc_test::capture_hash_word(hash, word);
              }
            }
            unsigned effective = rdna3 && ((policy & 4) || !(policy & 2)) ? m & ~12U : m;
            EXPECT_EQ(hash, goc_test::rounding_hardware_hashes[op][policy & 3][effective])
                << op << '/' << rdna3 << '/' << wide << '/' << policy << '/' << m;
          }
}

TEST(RoundingArchitecture, DppMasksAliasesAndValidation) {
  for (unsigned op = 0; op < 4; ++op)
    for (bool rdna3 : {false, true})
      for (bool wide : {false, true})
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
          for (unsigned policy = 0; policy < 8; ++policy)
            for (auto descriptor : goc_test::dpp_modes) {
              unsigned lanes = wide ? 64 : 32;
              uint64_t exec_mask = wide ? 0x9171abcd8912a5a5ULL : 0x8912a5a5U;
              uint64_t flags = cpu | (policy & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                               (policy & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) |
                               (policy & 4 ? GOC_FP_IEEE_MODE : 0);
              uint32_t mode = modifiers((policy * 5) % 32);
              uint32_t effective =
                  rdna3 && ((policy & 4) || !(policy & 2)) ? mode & ~GOC_ALU_OMOD_HALF : mode;
              uint32_t input[64], expected[64];
              for (unsigned lane = 0; lane < lanes; ++lane)
                input[lane] = goc_test::unary_capture_input(0, lane % 32);
              for (unsigned lane = 0; lane < lanes; ++lane) {
                expected[lane] = input[lane];
                int source;
                if (goc_test::dpp_source(descriptor, exec_mask, lane, source)) {
                  uint32_t word = source < 0 ? 0 : input[source];
                  if ((policy & 1) && !(word & 0x7f800000))
                    word &= 0x80000000;
                  expected[lane] =
                      goc::as_bits(goc_test::unary_reference(op, goc::as_float(word), effective));
                }
              }
              uint32_t state = 0x80000055;
              ASSERT_EQ(invoke(op, rdna3, wide, flags, exec_mask, mode | descriptor, input, input,
                               &state),
                        GOC_SUCCESS);
              EXPECT_EQ(state, 0x80000055U);
              for (unsigned lane = 0; lane < lanes; ++lane) {
                if ((expected[lane] & 0x7fffffff) > 0x7f800000) {
                  EXPECT_GT(input[lane] & 0x7fffffff, 0x7f800000U);
                } else {
                  EXPECT_EQ(input[lane], expected[lane]);
                }
              }
              EXPECT_EQ(invoke(op, rdna3, wide, flags, 0, 0, nullptr, nullptr, &state),
                        GOC_SUCCESS);
              EXPECT_EQ(invoke(op, rdna3, wide, flags, UINT64_MAX, GOC_ALU_NEG_B, nullptr, nullptr,
                               &state),
                        GOC_ERROR_INVALID_FLAGS);
              EXPECT_EQ(invoke(op, rdna3, wide, 2ULL << 16, 0, 0, nullptr, nullptr, &state),
                        GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
              EXPECT_EQ(state, 0x80000055U);
            }
}

TEST(RoundingArchitecture, ExactDppAliasesAndHostEnvironment) {
  goc_test::ScopedFpEnvironment environment;
  ASSERT_TRUE(environment.saved());
  for (unsigned op = 0; op < 4; ++op)
    for (bool rdna3 : {false, true})
      for (bool wide : {false, true})
        for (unsigned policy = 0; policy < 8; ++policy)
          for (unsigned m : {0U, 1U, 4U, 8U, 12U, 27U})
            for (auto descriptor : goc_test::dpp_modes) {
              unsigned lanes = wide ? 64 : 32;
              uint64_t exec_mask = wide ? 0x9192a5a571abcdefULL : 0x71abcdef;
              uint64_t flags = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT |
                               (policy & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                               (policy & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) |
                               (policy & 4 ? GOC_FP_IEEE_MODE : 0);
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
                  ASSERT_EQ(invoke(op, false, false, flags, 1, modifiers(effective), output, input,
                                   nullptr),
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
                  uint32_t input[64], state = 0x80000040;
                  std::memcpy(input, original, lanes * sizeof(uint32_t));
                  ASSERT_EQ(invoke(op, rdna3, wide, flags, exec_mask, modifiers(m) | descriptor,
                                   input, input, nullptr),
                            GOC_SUCCESS);
                  EXPECT_EQ(std::memcmp(input, expected, lanes * sizeof(uint32_t)), 0);
                  if (!rdna3) {
                    EXPECT_EQ(state, expected_state);
                  }
                  EXPECT_EQ(std::fegetround(), rounding);
                  // LLVM may lower vectorized integer shifts through FP conversions.
                  // Sticky flags may be raised, but controls and existing flags survive.
                  EXPECT_EQ(std::fetestexcept(FE_DIVBYZERO), FE_DIVBYZERO);
#if defined(__x86_64__) || defined(_M_X64)
                  EXPECT_EQ(_mm_getcsr() & ~0x3fU, controls & ~0x3fU);
#endif
                }
            }
}

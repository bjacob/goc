// SPDX-License-Identifier: MIT

#include "dpp_reference.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "half_reference.h"
#include "half_unary_reference.h"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdint.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace {

const uint64_t exact = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;

int invoke(unsigned op, bool rdna3, bool wide, uint64_t flags, uint64_t exec_mask, uint64_t mode,
           uint32_t *d, const uint32_t *a, uint32_t *state) {
  const decltype(&goc_v_trunc_f16) narrow[2][9] = {
      {goc_v_trunc_f16_rdna4, goc_v_ceil_f16_rdna4, goc_v_rndne_f16_rdna4, goc_v_floor_f16_rdna4,
       goc_v_frexp_mant_f16_rdna4, goc_v_fract_f16_rdna4, goc_v_rcp_f16_rdna4, goc_v_rsq_f16_rdna4,
       goc_v_sqrt_f16_rdna4},
      {goc_v_trunc_f16, goc_v_ceil_f16, goc_v_rndne_f16, goc_v_floor_f16, goc_v_frexp_mant_f16,
       goc_v_fract_f16, goc_v_rcp_f16, goc_v_rsq_f16, goc_v_sqrt_f16}};
  const decltype(&goc_v_trunc_f16_wave64) wave64[2][9] = {
      {goc_v_trunc_f16_rdna4_wave64, goc_v_ceil_f16_rdna4_wave64, goc_v_rndne_f16_rdna4_wave64,
       goc_v_floor_f16_rdna4_wave64, goc_v_frexp_mant_f16_rdna4_wave64,
       goc_v_fract_f16_rdna4_wave64, goc_v_rcp_f16_rdna4_wave64, goc_v_rsq_f16_rdna4_wave64,
       goc_v_sqrt_f16_rdna4_wave64},
      {goc_v_trunc_f16_wave64, goc_v_ceil_f16_wave64, goc_v_rndne_f16_wave64,
       goc_v_floor_f16_wave64, goc_v_frexp_mant_f16_wave64, goc_v_fract_f16_wave64,
       goc_v_rcp_f16_wave64, goc_v_rsq_f16_wave64, goc_v_sqrt_f16_wave64}};
  return wide ? wave64[rdna3][op](flags, exec_mask, mode, &d, &a, state)
              : narrow[rdna3][op](flags, uint32_t(exec_mask), mode, &d, &a, state);
}

uint32_t modifiers(unsigned m) {
  return (m & 1 ? GOC_ALU_NEG_A : 0) | (m & 2 ? GOC_ALU_ABS_A : 0) | ((m >> 2 & 3) << 6) |
         (m & 16 ? GOC_ALU_CLAMP : 0);
}

uint64_t policy_flags(unsigned policy) {
  return (policy & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
         (policy & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) | (policy & 4 ? GOC_FP_IEEE_MODE : 0);
}

void compare_half(uint32_t actual, uint32_t expected, unsigned shift) {
  EXPECT_EQ((actual ^ expected) & ~(65535U << shift), 0U);
  auto a = uint16_t(actual >> shift), e = uint16_t(expected >> shift);
  if ((e & 0x7fff) > 0x7c00)
    EXPECT_GT(a & 0x7fff, 0x7c00);
  else
    EXPECT_EQ(a, e);
}

} // namespace

TEST(HalfUnaryExact, ArchitecturePoliciesDppSelectorsAndAliases) {
  const uint16_t values[] = {0,      0x8000, 1,      0x8001, 0x3ff,  0x83ff, 0x400,  0x8400,
                             0x3800, 0xb800, 0x3bff, 0xbbff, 0x3c00, 0xbc00, 0x3e00, 0xbe00,
                             0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7c01, 0xfc01, 0x7e55, 0xfe55};
  for (unsigned op : {0U, 1U, 2U, 3U, 4U, 6U, 7U, 8U})
    for (bool rdna3 : {false, true})
      for (bool precise : {false, true})
        for (bool wide : {false, true})
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
            for (unsigned policy = 0; policy < 8; ++policy)
              for (unsigned m = 0; m < 32; ++m)
                for (auto dpp : goc_test::dpp_modes) {
                  unsigned lanes = wide ? 64 : 32, a_shift = m & 1 ? 16 : 0,
                           d_shift = m & 2 ? 16 : 0;
                  uint64_t exec_mask = wide ? 0x9192a5a571abcdefULL : 0x71abcdef;
                  uint64_t flags = cpu | policy_flags(policy) | (precise ? exact : 0);
                  uint32_t mode = modifiers(m) | (a_shift ? GOC_ALU_HIGH_A : 0) |
                                  (d_shift ? GOC_ALU_HIGH_D : 0);
                  uint32_t effective =
                      rdna3 && ((policy & 4) || !(policy & 2)) ? mode & ~GOC_ALU_OMOD_HALF : mode;
                  uint32_t input[64], expected[64], state = 0x80000040, want_state = state;
                  for (unsigned lane = 0; lane < lanes; ++lane)
                    expected[lane] = input[lane] =
                        (uint32_t(values[lane % 24]) << a_shift) | (0x1234U << (16 - a_shift));
                  for (unsigned lane = 0; lane < lanes; ++lane) {
                    int source;
                    if (!goc_test::dpp_source(dpp, exec_mask, lane, source))
                      continue;
                    uint32_t word = source < 0 ? 0 : input[source];
                    uint16_t half = uint16_t(word >> a_shift);
                    if (op != 4 && (policy & 1) && (half & 0x7fff) < 0x400) {
                      word &= ~(65535U << a_shift);
                      word |= uint32_t(half & 0x8000) << a_shift;
                    }
                    const unsigned reference_ops[] = {0, 1, 2, 3, 10, 9, 5, 6, 4};
                    uint16_t result =
                        goc_test::half_unary_reference(reference_ops[op], word, effective, false);
                    if ((op == 5 || op >= 6) && (policy & 2) && (result & 0x7fff) < 0x400)
                      result &= 0x8000;
                    expected[lane] =
                        (expected[lane] & ~(65535U << d_shift)) | (uint32_t(result) << d_shift);
                  }
                  ASSERT_EQ(invoke(op, rdna3, wide, flags, exec_mask, mode | dpp, input, input,
                                   precise ? nullptr : &state),
                            GOC_SUCCESS);
                  EXPECT_EQ(state, want_state);
                  for (unsigned lane = 0; lane < lanes; ++lane)
                    compare_half(input[lane], expected[lane], d_shift);
                }
}

TEST(HalfUnaryExact, HostControlsAndValidation) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (unsigned op : {0U, 1U, 2U, 3U, 4U, 6U, 7U, 8U})
    for (bool rdna3 : {false, true})
      for (bool wide : {false, true}) {
        uint32_t input[64], baseline[64], actual[64];
        for (unsigned lane = 0; lane < 64; ++lane)
          input[lane] = lane * 0x7395a831U;
        for (unsigned policy = 0; policy < 8; ++policy)
          for (unsigned m = 0; m < 32; ++m) {
            uint64_t flags = exact | policy_flags(policy) | (m & 1 ? GOC_FP16_OVFL : 0);
            uint32_t mode = modifiers(m) | GOC_ALU_HIGH_D;
            uint32_t expected_state = 0x80000040;
            std::fill(baseline, baseline + 64, 0xdeadbeef);
            ASSERT_EQ(invoke(op, rdna3, wide, flags, UINT64_MAX, mode, baseline, input, nullptr),
                      GOC_SUCCESS);
            for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO})
              for (unsigned denorm = 0; denorm < 4; ++denorm) {
                ASSERT_EQ(std::fesetround(rounding), 0);
#if defined(__x86_64__) || defined(_M_X64)
                unsigned controls = (_mm_getcsr() & ~0x8040U) | ((denorm & 1) ? 0x40U : 0) |
                                    ((denorm & 2) ? 0x8000U : 0);
                _mm_setcsr(controls);
#endif
                uint32_t state = 0x80000040;
                std::fill(actual, actual + 64, 0xdeadbeef);
                ASSERT_EQ(invoke(op, rdna3, wide, flags, UINT64_MAX, mode, actual, input, nullptr),
                          GOC_SUCCESS);
                EXPECT_EQ(state, expected_state);
                EXPECT_EQ(std::fegetround(), rounding);
#if defined(__x86_64__) || defined(_M_X64)
                EXPECT_EQ(_mm_getcsr(), controls);
#endif
                EXPECT_TRUE(std::equal(actual, actual + 64, baseline));
              }
          }
        uint32_t state = 0x12345678;
        EXPECT_EQ(invoke(op, rdna3, wide, exact, 0, 0, nullptr, nullptr, nullptr), GOC_SUCCESS);
        EXPECT_EQ(invoke(op, rdna3, wide, exact, 0, GOC_ALU_NEG_B, nullptr, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(invoke(op, rdna3, wide, exact, 0, 1ULL << 63, nullptr, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(invoke(op, rdna3, wide, 2 * GOC_SEMANTICS_EXACT_EMPIRICAL, 0, 0, nullptr, nullptr,
                         &state),
                  GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
        if (rdna3) {
          EXPECT_EQ(invoke(op, true, wide, exact, 0, 0, nullptr, nullptr, &state),
                    GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
        }
        EXPECT_EQ(state, 0x12345678U);
      }
}

TEST(HalfUnaryExact, SfuLooseGuestModesAndOptionalState) {
  const unsigned reference_ops[] = {5, 6, 4};
  const unsigned cpu_count = unsigned(goc_init_cpu_flags() + 1);
  for (unsigned op = 6; op < 9; ++op)
    for (bool saturate : {false, true})
      for (unsigned policy = 0; policy < 4; ++policy)
        for (unsigned m = 0; m < 32; ++m)
          for (unsigned base = 0; base < 65536; base += 32) {
            uint32_t input[32], output[32], state = 0x8badf00d;
            uint32_t mode = modifiers(m) | GOC_ALU_HIGH_A | GOC_ALU_HIGH_D;
            for (unsigned lane = 0; lane < 32; ++lane) {
              input[lane] = ((base + lane) << 16) | 0x7c01;
              output[lane] = 0xdeadbeef;
            }
            uint64_t flags =
                (base / 32 % cpu_count) | policy_flags(policy) | (saturate ? GOC_FP16_OVFL : 0);
            ASSERT_EQ(invoke(op, false, false, flags, UINT32_MAX, mode, output, input, &state),
                      GOC_SUCCESS);
            ASSERT_EQ(state, 0x8badf00dU);
            for (unsigned lane = 0; lane < 32; ++lane) {
              uint16_t half = uint16_t(base + lane);
              if ((policy & 1) && (half & 0x7fff) < 0x400)
                half &= 0x8000;
              uint16_t expected = goc_test::half_unary_reference(
                  reference_ops[op - 6], uint32_t(half) << 16, mode, saturate);
              if ((policy & 2) && (expected & 0x7fff) < 0x400)
                expected &= 0x8000;
              ASSERT_EQ(output[lane] & 0xffff, 0xbeefU);
              uint16_t actual = uint16_t(output[lane] >> 16);
              if ((expected & 0x7fff) > 0x7c00) {
                ASSERT_GT(actual & 0x7fff, 0x7c00);
              } else if ((expected & 0x7fff) && (expected & 0x7fff) < 0x7c00) {
                ASSERT_EQ(actual & 0x8000, expected & 0x8000);
                ASSERT_LE(std::abs(int(actual) - int(expected)), 1);
              } else {
                ASSERT_EQ(actual, expected);
              }
            }
          }
}

TEST(HalfUnaryExact, SfuRdna3NanQuieting) {
  for (unsigned op = 6; op < 9; ++op)
    for (bool rdna3 : {false, true})
      for (bool ieee : {false, true})
        for (bool wide : {false, true})
          for (unsigned m = 0; m < 32; ++m) {
            uint32_t input[64], output[64];
            for (unsigned lane = 0; lane < 64; ++lane) {
              input[lane] = ((0x7c01U | (lane * 0x1357U & 0x83fe)) << 16) | 0x1234;
              output[lane] = 0xdeadbeef;
            }
            ASSERT_EQ(invoke(op, rdna3, wide, exact | (ieee ? GOC_FP_IEEE_MODE : 0), UINT64_MAX,
                             modifiers(m) | GOC_ALU_HIGH_A, output, input, nullptr),
                      GOC_SUCCESS);
            for (unsigned lane = 0; lane < (wide ? 64U : 32U); ++lane) {
              uint16_t expected = uint16_t(input[lane] >> 16);
              if (m & 2)
                expected &= 0x7fff;
              if (m & 1)
                expected ^= 0x8000;
              if (!rdna3 || ieee)
                expected |= 0x200;
              if (m & 16)
                expected = 0;
              EXPECT_EQ(output[lane], 0xdead0000U | expected);
            }
          }
}

TEST(HalfUnaryExact, RoundingNanPayloadsIgnoreIeeeMode) {
  // rocjitsu util::rndne_scalar and ceil/floor/trunc_scalar quiet signaling
  // NaNs unconditionally. RDNA3 IEEE mode does not gate this family's quieting.
  const uint32_t inputs[] = {0x7c01, 0xfc55, 0x7e23, 0xfe7f};
  const uint32_t results[] = {0x7e01, 0xfe55, 0x7e23, 0xfe7f};
  for (unsigned op = 0; op < 4; ++op)
    for (bool rdna3 : {false, true})
      for (bool wide : {false, true})
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
          for (unsigned policy = 0; policy < 8; ++policy) {
            uint64_t flags = cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT |
                             (policy & 1 ? GOC_FP_IEEE_MODE : 0) |
                             (policy & 2 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                             (policy & 4 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0);
            uint32_t a[64], d[64];
            for (unsigned lane = 0; lane < 64; ++lane) {
              a[lane] = inputs[lane % 4] << 16;
              d[lane] = 0x12345678;
            }
            uint64_t exec_mask = wide ? 0x800000010000000fULL : 0x8000000fU;
            ASSERT_EQ(invoke(op, rdna3, wide, flags, exec_mask, GOC_ALU_HIGH_A | GOC_ALU_HIGH_D, d,
                             a, nullptr),
                      GOC_SUCCESS);
            for (unsigned lane = 0; lane < 64; ++lane) {
              bool active = (exec_mask >> lane) & 1;
              EXPECT_EQ(d[lane], active ? (results[lane % 4] << 16) | 0x5678U : 0x12345678U)
                  << op << '/' << rdna3 << '/' << wide << '/' << policy << '/' << lane;
            }
          }
}

TEST(HalfUnaryExact, FrexpEveryEncodingAndDenormalPolicy) {
  // rocjitsu widens to FP32 before FREXP: even the smallest half subnormal
  // is normal at that width. Neither input nor output flushing affects it.
  for (bool rdna3 : {false, true})
    for (bool wide : {false, true})
      for (bool precise : {false, true})
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
          for (unsigned policy = 0; policy < 4; ++policy)
            for (unsigned halves = 0; halves < 4; ++halves)
              for (unsigned base = 0; base < 65536; base += wide ? 64 : 32) {
                const unsigned lanes = wide ? 64 : 32;
                const unsigned a_shift = halves & 1 ? 16 : 0, d_shift = halves & 2 ? 16 : 0;
                uint32_t input[64], expected[64];
                for (unsigned lane = 0; lane < lanes; ++lane) {
                  const uint16_t half = uint16_t(base + lane);
                  input[lane] = (uint32_t(half) << a_shift) | (0x1234U << (16 - a_shift));
                  uint16_t result;
                  if ((half & 0x7fff) >= 0x7c00)
                    result = half | ((half & 0x7fff) > 0x7c00 ? 0x200 : 0);
                  else {
                    int exponent;
                    result = goc_test::half_bits(std::frexp(goc_test::half_value(half), &exponent));
                  }
                  expected[lane] =
                      (input[lane] & ~(65535U << d_shift)) | (uint32_t(result) << d_shift);
                }
                const uint64_t flags = cpu | policy_flags(policy) | (precise ? exact : 0);
                const uint64_t mode =
                    (a_shift ? GOC_ALU_HIGH_A : 0) | (d_shift ? GOC_ALU_HIGH_D : 0);
                ASSERT_EQ(invoke(4, rdna3, wide, flags, UINT64_MAX, mode, input, input, nullptr),
                          GOC_SUCCESS);
                for (unsigned lane = 0; lane < lanes; ++lane) {
                  if (precise)
                    ASSERT_EQ(input[lane], expected[lane]) << base + lane << '/' << policy;
                  else
                    compare_half(input[lane], expected[lane], d_shift);
                }
              }
}

TEST(HalfUnaryExact, SfuLiteralDenormalOverflowAndSpecialValues) {
  // Finite expectations use powers of four, so the borrowed approximation has
  // an exactly representable result (apart from reciprocal overflow).
  const uint16_t inputs[] = {0,      1,      0x0400, 0x3c00, 0x4400, 0x7c00,
                             0x8000, 0x8001, 0xbc00, 0xfc00, 0x7c01, 0xfe23};
  const uint16_t results[3][12] = {
      {0x7c00, 0x7c00, 0x7400, 0x3c00, 0x3400, 0, 0xfc00, 0xfc00, 0xbc00, 0x8000, 0x7e01, 0xfe23},
      {0x7c00, 0x6c00, 0x5800, 0x3c00, 0x3800, 0, 0xfc00, 0xfe00, 0xfe00, 0xfe00, 0x7e01, 0xfe23},
      {0, 0x0c00, 0x2000, 0x3c00, 0x4000, 0x7c00, 0x8000, 0xfe00, 0xfe00, 0xfe00, 0x7e01, 0xfe23}};
  for (unsigned op = 0; op < 3; ++op)
    for (bool rdna3 : {false, true})
      for (bool wide : {false, true})
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
          for (unsigned policy = 0; policy < 8; ++policy)
            for (bool saturate : {false, true}) {
              const uint64_t flags =
                  cpu | exact | policy_flags(policy) | (saturate ? GOC_FP16_OVFL : 0);
              const uint64_t exec_mask = wide ? 0x9192a5a571abcdefULL : 0x71abcdef;
              uint32_t words[64], expected[64];
              for (unsigned lane = 0; lane < 64; ++lane) {
                unsigned i = lane % 12;
                expected[lane] = words[lane] = (uint32_t(inputs[i]) << 16) | 0x1234U;
                if (!((exec_mask >> lane) & 1))
                  continue;
                unsigned source = (policy & 1) && (i == 1 || i == 7) ? i - 1 : i;
                uint16_t result = results[op][source];
                if (rdna3 && !(policy & 4) && i == 10)
                  result = 0x7c01;
                if (saturate && (inputs[i] & 0x7fff) < 0x7c00 && (result & 0x7fff) == 0x7c00)
                  result = (result & 0x8000) | 0x7bff;
                expected[lane] = (uint32_t(result) << 16) | 0x1234U;
              }
              ASSERT_EQ(invoke(op + 6, rdna3, wide, flags, exec_mask,
                               GOC_ALU_HIGH_A | GOC_ALU_HIGH_D, words, words, nullptr),
                        GOC_SUCCESS);
              EXPECT_TRUE(std::equal(words, words + 64, expected))
                  << op << '/' << rdna3 << '/' << wide << '/' << policy << '/' << saturate;

              if (op == 0) {
                // 1 / 32768 is a half subnormal: output policy, not input
                // policy, decides whether that exactly representable result survives.
                std::fill(words, words + 64, 0x12347800U);
                ASSERT_EQ(invoke(6, rdna3, wide, flags, 1, 0, words, words, nullptr), GOC_SUCCESS);
                EXPECT_EQ(words[0], policy & 2 ? 0x12340000U : 0x12340200U);
                EXPECT_TRUE(std::all_of(words + 1, words + 64,
                                        [](uint32_t value) { return value == 0x12347800U; }));
              }
            }
}

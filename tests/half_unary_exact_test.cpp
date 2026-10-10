// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "dpp_reference.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "half_unary_exact_hardware.h"
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
  const decltype(&goc_v_trunc_f16) narrow[2][6] = {
      {goc_v_trunc_f16_rdna4, goc_v_ceil_f16_rdna4, goc_v_rndne_f16_rdna4, goc_v_floor_f16_rdna4,
       goc_v_frexp_mant_f16_rdna4, goc_v_fract_f16_rdna4},
      {goc_v_trunc_f16, goc_v_ceil_f16, goc_v_rndne_f16, goc_v_floor_f16, goc_v_frexp_mant_f16,
       goc_v_fract_f16}};
  const decltype(&goc_v_trunc_f16_wave64) wave64[2][6] = {
      {goc_v_trunc_f16_rdna4_wave64, goc_v_ceil_f16_rdna4_wave64, goc_v_rndne_f16_rdna4_wave64,
       goc_v_floor_f16_rdna4_wave64, goc_v_frexp_mant_f16_rdna4_wave64,
       goc_v_fract_f16_rdna4_wave64},
      {goc_v_trunc_f16_wave64, goc_v_ceil_f16_wave64, goc_v_rndne_f16_wave64,
       goc_v_floor_f16_wave64, goc_v_frexp_mant_f16_wave64, goc_v_fract_f16_wave64}};
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

// Independently checked against every captured raw EXCP_FLAG_USER word.
uint32_t exception(unsigned op, uint16_t input, unsigned policy, unsigned modifier) {
  if (modifier & 16)
    return 0;
  if (modifier & 2)
    input &= 0x7fff;
  if (modifier & 1)
    input ^= 0x8000;
  unsigned magnitude = input & 0x7fff;
  if ((magnitude > 0x7c00 && !(magnitude & 0x200)) || (op >= 4 && magnitude == 0x7c00))
    return GOC_EXCEPTION_INVALID;
  uint32_t flags = magnitude && magnitude < 0x400 && !(policy & 1) ? GOC_EXCEPTION_INPUT_DENORM : 0;
  if (op == 5 && magnitude < 0x7c00 && !(modifier & 12)) {
    double x = goc_test::half_value(input);
    if ((policy & 1) && magnitude < 0x400)
      x = 0;
    double fraction = x - std::floor(x);
    double rounded = goc_test::half_value(goc_test::half_bits(fraction, false));
    if (rounded != fraction)
      flags |= GOC_EXCEPTION_INEXACT;
    if ((policy & 2) && rounded > 0 && rounded < 0x1p-14)
      flags |= GOC_EXCEPTION_UNDERFLOW | GOC_EXCEPTION_INEXACT;
  }
  return flags;
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

TEST(HalfUnaryExact, ExhaustiveHardwareResultsAndWaveFlags) {
  const unsigned cpu_count = unsigned(goc_init_cpu_flags() + 1);
  for (unsigned op = 0; op < 6; ++op)
    for (bool precise : {false, true})
      for (unsigned saturate = 0; saturate < 2; ++saturate)
        for (unsigned policy = 0; policy < 4; ++policy)
          for (unsigned m = 0; m < 32; ++m) {
            bool wide = m & 1;
            unsigned lanes = wide ? 64 : 32;
            uint64_t hash = goc_test::capture_hash_seed, pair_hash = hash;
            for (unsigned base = 0; base < 65536; base += lanes) {
              unsigned a_shift = (base / lanes & 1) ? 16 : 0;
              unsigned d_shift = (base / lanes & 2) ? 16 : 0;
              uint32_t mode =
                  modifiers(m) | (a_shift ? GOC_ALU_HIGH_A : 0) | (d_shift ? GOC_ALU_HIGH_D : 0);
              uint64_t flags = (base / lanes % cpu_count) | policy_flags(policy) |
                               (saturate ? GOC_FP16_OVFL : 0) | (precise ? exact : 0);
              uint32_t input[64], output[64], state = 0x80000000, expected_state = state;
              for (unsigned lane = 0; lane < lanes; ++lane) {
                input[lane] = ((base + lane) << a_shift) | (0x7c01U << (16 - a_shift));
                output[lane] = 0xdeadbeef;
                expected_state |= exception(op, uint16_t(base + lane), policy, m);
              }
              ASSERT_EQ(invoke(op, false, wide, flags, UINT64_MAX, mode, output, input, &state),
                        GOC_SUCCESS);
              ASSERT_EQ(state, precise ? expected_state : 0x80000000U);
              for (unsigned lane = 0; lane < lanes; ++lane) {
                ASSERT_EQ((output[lane] ^ 0xdeadbeefU) & ~(65535U << d_shift), 0U);
                uint16_t word = uint16_t(output[lane] >> d_shift);
                if (!precise && (word & 0x7fff) > 0x7c00) {
                  // Loose semantics permit arbitrary NaN payloads. Reconstitute
                  // the captured payload only for hashing the remaining numeric bits.
                  word = uint16_t(base + lane);
                  if (m & 2)
                    word &= 0x7fff;
                  if (m & 1)
                    word ^= 0x8000;
                  word = (word & 0x7fff) == 0x7c00 ? 0xfe00 : word | 0x200;
                }
                hash = goc_test::capture_hash_word(hash, word);
                pair_hash = goc_test::capture_hash_word(pair_hash, word);
                pair_hash = goc_test::capture_hash_word(
                    pair_hash, exception(op, uint16_t(base + lane), policy, m));
              }
            }
            EXPECT_EQ(hash,
                      (op == 5 ? goc_test::half_fract_exact_hashes[policy][m][0]
                               : goc_test::half_unary_exact_hashes[op][saturate][policy & 1][m][0]))
                << op << '/' << precise << '/' << saturate << '/' << policy << '/' << m;
            EXPECT_EQ(pair_hash,
                      (op == 5
                           ? goc_test::half_fract_exact_hashes[policy][m][1]
                           : goc_test::half_unary_exact_hashes[op][saturate][policy & 1][m][1]));
          }
}

TEST(HalfUnaryExact, OnlyParticipatingLanesReport) {
  for (unsigned op = 0; op < 6; ++op)
    for (unsigned policy = 0; policy < 4; ++policy)
      for (bool clamp : {false, true})
        for (unsigned value = 0; value < 65536; ++value) {
          bool wide = value & 1;
          unsigned lanes = wide ? 64 : 32, lane = (value >> 1) % lanes;
          uint32_t input[64], output[64];
          std::fill(input, input + 64, 0x7c01); // Inactive signaling NaNs must not report.
          std::fill(output, output + 64, 0xdeadbeef);
          input[lane] = value;
          uint32_t state = 0x80000040;
          ASSERT_EQ(invoke(op, false, wide, exact | policy_flags(policy), 1ULL << lane,
                           clamp ? GOC_ALU_CLAMP : 0, output, input, &state),
                    GOC_SUCCESS);
          ASSERT_EQ(state, 0x80000040U | exception(op, uint16_t(value), policy, clamp ? 16 : 0));
          for (unsigned i = 0; i < lanes; ++i)
            if (i != lane) {
              ASSERT_EQ(output[i], 0xdeadbeefU);
            }
        }
}

TEST(HalfUnaryExact, ArchitecturePoliciesDppSelectorsAndAliases) {
  const uint16_t values[] = {0,      0x8000, 1,      0x8001, 0x3ff,  0x83ff, 0x400,  0x8400,
                             0x3800, 0xb800, 0x3bff, 0xbbff, 0x3c00, 0xbc00, 0x3e00, 0xbe00,
                             0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7c01, 0xfc01, 0x7e55, 0xfe55};
  for (unsigned op = 0; op < 6; ++op)
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
                    if (precise && !rdna3)
                      want_state |= exception(op, half, policy, m);
                    if ((policy & 1) && (half & 0x7fff) < 0x400) {
                      word &= ~(65535U << a_shift);
                      word |= uint32_t(half & 0x8000) << a_shift;
                    }
                    uint16_t result = goc_test::half_unary_reference(op == 4   ? 10
                                                                     : op == 5 ? 9
                                                                               : op,
                                                                     word, effective, false);
                    if (op == 5 && (policy & 2) && result < 0x400)
                      result = 0;
                    expected[lane] =
                        (expected[lane] & ~(65535U << d_shift)) | (uint32_t(result) << d_shift);
                  }
                  ASSERT_EQ(invoke(op, rdna3, wide, flags, exec_mask, mode | dpp, input, input,
                                   rdna3 && precise ? nullptr : &state),
                            GOC_SUCCESS);
                  EXPECT_EQ(state, want_state);
                  for (unsigned lane = 0; lane < lanes; ++lane)
                    compare_half(input[lane], expected[lane], d_shift);
                }
}

TEST(HalfUnaryExact, HostControlsAndValidation) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (unsigned op = 0; op < 6; ++op)
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
            ASSERT_EQ(invoke(op, rdna3, wide, flags, UINT64_MAX, mode, baseline, input,
                             rdna3 ? nullptr : &expected_state),
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
                ASSERT_EQ(invoke(op, rdna3, wide, flags, UINT64_MAX, mode, actual, input,
                                 rdna3 ? nullptr : &state),
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
        EXPECT_EQ(invoke(op, rdna3, wide, exact, 0, 0, nullptr, nullptr, rdna3 ? nullptr : &state),
                  GOC_SUCCESS);
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

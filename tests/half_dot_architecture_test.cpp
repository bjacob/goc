// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "dpp_reference.h"
#include "fp_environment.h"
#include "goc/goc.h"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace {

using Fn32 = decltype(&goc_v_dot2_f16_f16);
using Fn64 = decltype(&goc_v_dot2_f16_f16_wave64);
const Fn32 functions[2][2] = {{goc_v_dot2_f16_f16, goc_v_dot2_bf16_bf16},
                              {goc_v_dot2_f16_f16_rdna4, goc_v_dot2_bf16_bf16_rdna4}};
const Fn64 wide_functions[2][2] = {
    {goc_v_dot2_f16_f16_wave64, goc_v_dot2_bf16_bf16_wave64},
    {goc_v_dot2_f16_f16_rdna4_wave64, goc_v_dot2_bf16_bf16_rdna4_wave64}};

uint64_t policy(unsigned guest) {
  return (guest & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
         (guest & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) | (guest & 4 ? GOC_FP_IEEE_MODE : 0);
}

int call(unsigned arch, bool wide, unsigned brain, uint64_t flags, uint64_t exec_mask,
         uint64_t mode, uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
         const uint32_t *const *c, uint32_t *state = nullptr) {
  if (wide)
    return wide_functions[arch][brain](flags, exec_mask, mode, d, a, b, c, state);
  return functions[arch][brain](flags, uint32_t(exec_mask), mode, d, a, b, c, state);
}

// Encode the small integral reference inputs/results exactly in either format.
uint16_t encode(int value, bool brain) {
  if (!value)
    return 0;
  int exponent;
  double mantissa = std::frexp(double(std::abs(value)), &exponent) * 2;
  unsigned fraction = brain ? 7 : 10, bias = brain ? 127 : 15;
  return uint16_t((value < 0 ? 0x8000 : 0) | ((exponent - 1 + bias) << fraction) |
                  unsigned(std::ldexp(mantissa - 1, fraction)));
}

} // namespace

TEST(HalfDotArchitecture, FixedPoliciesModifiersDppMasksAliasesAndGuards) {
  for (unsigned brain = 0; brain < 2; ++brain)
    for (bool wide : {false, true})
      for (unsigned guest = 0; guest < 8; ++guest)
        for (unsigned variant = 0; variant < 256; ++variant) {
          unsigned mode = (variant & 63) | (variant & 64 ? GOC_ALU_HIGH_C : 0) |
                          (variant & 128 ? GOC_ALU_HIGH_D : 0);
          unsigned shift = mode & GOC_ALU_HIGH_D ? 16 : 0, target = (variant / 2) % 4;
          bool shared = variant & 1;
          uint64_t descriptor = variant % 8 ? goc_test::dpp_modes[variant % 8 - 1] : 0;
          if (wide && (descriptor & GOC_DPP16))
            descriptor =
                (descriptor & ~GOC_DPP_ROW_MASK) | (uint64_t(variant & 15) << GOC_DPP_ROW_SHIFT);
          uint64_t exec_mask = variant & 2 ? 0x87654321fedcba98ULL : UINT64_MAX;
          if (!wide)
            exec_mask = uint32_t(exec_mask);
          uint32_t initial[4][66], words[4][66], expected[4][66];
          int numbers[3][64][2];
          for (unsigned reg = 0; reg < 4; ++reg)
            std::fill_n(initial[reg], 66, 0x12345678U);
          for (unsigned reg = 0; reg < 3; ++reg)
            for (unsigned lane = 0; lane < 64; ++lane) {
              for (unsigned half = 0; half < 2; ++half)
                numbers[reg][lane][half] = int((lane * (2 * reg + 1) + 3 * half) % 9) - 4;
              initial[reg][lane + 1] = encode(numbers[reg][lane][0], brain) |
                                       uint32_t(encode(numbers[reg][lane][1], brain)) << 16;
            }
          std::memcpy(expected, initial, sizeof(expected));
          for (unsigned lane = 0; lane < (wide ? 64U : 32U); ++lane) {
            int source = int(lane);
            bool active = descriptor ? goc_test::dpp_source(descriptor, exec_mask, lane, source)
                                     : ((exec_mask >> lane) & 1);
            if (!active)
              continue;
            int operands[3][2];
            for (unsigned reg = 0; reg < 3; ++reg)
              for (unsigned half = 0; half < 2; ++half) {
                int value = reg == 0 ? (source < 0 ? 0 : numbers[0][source][half])
                                     : numbers[shared ? 0 : reg][lane][half];
                if (mode & (GOC_ALU_ABS_A << reg))
                  value = std::abs(value);
                if (mode & (GOC_ALU_NEG_A << reg))
                  value = -value;
                operands[reg][half] = value;
              }
            int value = operands[0][0] * operands[1][0] + operands[0][1] * operands[1][1] +
                        operands[2][bool(mode & GOC_ALU_HIGH_C)];
            expected[target][lane + 1] = (initial[target][lane + 1] & ~(0xffffU << shift)) |
                                         uint32_t(encode(value, brain)) << shift;
          }
          for (unsigned arch = 0; arch < 2; ++arch)
            for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
              std::memcpy(words, initial, sizeof(words));
              const uint32_t *a[] = {words[0] + 1}, *b[] = {words[shared ? 0 : 1] + 1},
                             *c[] = {words[shared ? 0 : 2] + 1};
              uint32_t *d[] = {words[target] + 1}, state = 0x12345678;
              ASSERT_EQ(call(arch, wide, brain, cpu | policy(guest), exec_mask, descriptor | mode,
                             d, a, b, c, &state),
                        GOC_SUCCESS);
              ASSERT_EQ(state, 0x12345678U);
              ASSERT_EQ(std::memcmp(words, expected, sizeof(words)), 0)
                  << brain << "/" << wide << "/" << guest << "/" << variant << "/" << arch << "/"
                  << cpu;
            }
        }
}

TEST(HalfDotArchitecture, EveryAccumulatorEncodingIgnoresGuestDenormalModes) {
  for (unsigned brain = 0; brain < 2; ++brain)
    for (unsigned guest = 0; guest < 8; ++guest)
      for (unsigned base = 0; base < 65536; base += 64) {
        uint32_t zero[64]{}, accumulator[64], result[64];
        for (unsigned lane = 0; lane < 64; ++lane)
          accumulator[lane] = (base + lane) * 0x10001U;
        const uint32_t *a[] = {zero}, *c[] = {accumulator};
        uint32_t *d[] = {result};
        for (unsigned arch = 0; arch < 2; ++arch)
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
            std::fill_n(result, 64, 0x12345678U);
            ASSERT_EQ(call(arch, true, brain, cpu | policy(guest), UINT64_MAX,
                           GOC_ALU_HIGH_C | GOC_ALU_HIGH_D, d, a, a, c),
                      GOC_SUCCESS);
            for (unsigned lane = 0; lane < 64; ++lane) {
              unsigned raw = base + lane, magnitude = raw & 0x7fff,
                       infinity = brain ? 0x7f80 : 0x7c00;
              ASSERT_EQ(result[lane] & 0xffff, 0x5678U);
              if (magnitude > infinity)
                ASSERT_GT((result[lane] >> 16) & 0x7fff, infinity);
              else {
                unsigned want = !magnitude || (brain && magnitude < 128) ? 0 : raw;
                ASSERT_EQ(result[lane] >> 16, want)
                    << brain << "/" << guest << "/" << raw << "/" << arch << "/" << cpu;
              }
            }
          }
      }
}

TEST(HalfDotArchitecture, LiteralPrecisionOverflowAndFixedDenormalPolicy) {
  struct Case {
    unsigned brain;
    uint32_t a, b, c;
    uint16_t result, saturated;
  };

  const Case cases[] = {{0, 1, 0x3c00, 0, 1, 1},
                        {0, 3, 0x3800, 0, 2, 2},
                        {0, 0x3c00, 0x3c00, 0x1000, 0x3c00, 0x3c00},
                        {0, 0x3c01, 0x3c00, 0x1000, 0x3c02, 0x3c02},
                        {0, 0x7bff, 0x4000, 0, 0x7c00, 0x7bff},
                        {0, 0xfbff, 0x4000, 0, 0xfc00, 0xfbff},
                        {0, 0x7c00, 0x3c00, 0, 0x7c00, 0x7c00},
                        {1, 1, 0x7f00, 0, 0, 0},
                        {1, 0x80, 0x3f00, 0, 0, 0},
                        {1, 0x7f7f, 0x4000, 0, 0x7f80, 0x7f80},
                        {1, 0x3f80, 0x3f80, 0x3b80, 0x3f80, 0x3f80}};
  for (const auto &v : cases)
    for (unsigned guest = 0; guest < 8; ++guest)
      for (bool saturate : {false, true}) {
        uint32_t a0[64], b0[64], c0[64], result[64];
        std::fill_n(a0, 64, v.a);
        std::fill_n(b0, 64, v.b);
        std::fill_n(c0, 64, v.c);
        const uint32_t *a[] = {a0}, *b[] = {b0}, *c[] = {c0};
        uint32_t *d[] = {result};
        for (unsigned arch = 0; arch < 2; ++arch)
          for (bool wide : {false, true})
            for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
              std::fill_n(result, 64, 0xabcd0000U);
              ASSERT_EQ(call(arch, wide, v.brain,
                             cpu | policy(guest) | (saturate ? GOC_FP16_OVFL : 0), UINT64_MAX, 0, d,
                             a, b, c),
                        GOC_SUCCESS);
              for (unsigned lane = 0; lane < (wide ? 64U : 32U); ++lane)
                ASSERT_EQ(result[lane], 0xabcd0000U | (saturate ? v.saturated : v.result));
            }
      }
}

TEST(HalfDotArchitecture, ReportingErrorsAndHostControls) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  uint32_t input[64], output[64];
  std::fill_n(input, 64, 0x3c013c03U);
  const uint32_t *a[] = {input};
  uint32_t *d[] = {output};
  for (unsigned arch = 0; arch < 2; ++arch)
    for (bool wide : {false, true})
      for (unsigned brain = 0; brain < 2; ++brain) {
        uint32_t state = 0x12345678;
        for (uint64_t mode : {uint64_t(0), goc_test::dpp_modes[0]}) {
          for (uint64_t semantics : {GOC_SEMANTICS_EXACT_EMPIRICAL,
                                     GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT})
            for (uint64_t exec_mask : {0ULL, ~0ULL})
              EXPECT_EQ(call(arch, wide, brain, semantics, exec_mask, mode, nullptr, nullptr,
                             nullptr, nullptr, &state),
                        GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
          EXPECT_EQ(call(arch, wide, brain, GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0,
                         mode, nullptr, nullptr, nullptr, nullptr),
                    GOC_ERROR_UNSUPPORTED_SEMANTICS);
          EXPECT_EQ(call(arch, wide, brain, 0, 0, mode, nullptr, nullptr, nullptr, nullptr, &state),
                    GOC_SUCCESS);
        }
        EXPECT_EQ(state, 0x12345678U);
        for (uint32_t invalid : {GOC_ALU_CLAMP, GOC_ALU_OMOD_2, GOC_ALU_HIGH_A, GOC_ALU_HIGH_B})
          EXPECT_EQ(call(arch, wide, brain, 0, 0, invalid, nullptr, nullptr, nullptr, nullptr),
                    GOC_ERROR_INVALID_FLAGS);
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
          for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO})
            for (unsigned denorm = 0; denorm < 4; ++denorm) {
              ASSERT_EQ(std::fesetround(rounding), 0);
#if defined(__x86_64__) || defined(_M_X64)
              _mm_setcsr((_mm_getcsr() & ~0x8040U) | (denorm & 1 ? 0x40U : 0) |
                         (denorm & 2 ? 0x8000U : 0));
              unsigned controls = _mm_getcsr() & ~0x3fU;
#endif
              std::fill_n(output, 64, 0xabcd0000U);
              ASSERT_EQ(call(arch, wide, brain, cpu | policy(7), UINT64_MAX, 0, d, a, a, a, &state),
                        GOC_SUCCESS);
              EXPECT_EQ(state, 0x12345678U);
              EXPECT_EQ(std::fegetround(), rounding);
#if defined(__x86_64__) || defined(_M_X64)
              EXPECT_EQ(_mm_getcsr() & ~0x3fU, controls);
#endif
            }
      }
}

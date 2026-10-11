// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "dpp_reference.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "mullit_checks.h"
#include "mullit_hardware.h"
#include "mullit_reference.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace {

uint64_t policy(unsigned guest) {
  return (guest & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
         (guest & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) | (guest & 4 ? GOC_FP_IEEE_MODE : 0);
}

int call(bool rdna4, bool wide, uint64_t flags, uint64_t exec_mask, uint64_t mode,
         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
         const uint32_t *const *c, uint32_t *state = nullptr) {
  if (wide)
    return (rdna4 ? goc_v_mullit_f32_rdna4_wave64 : goc_v_mullit_f32_wave64)(flags, exec_mask, mode,
                                                                             d, a, b, c, state);
  return (rdna4 ? goc_v_mullit_f32_rdna4 : goc_v_mullit_f32)(flags, uint32_t(exec_mask), mode, d, a,
                                                             b, c, state);
}

} // namespace

TEST(MullitArchitecture, PoliciesAllModifiersDppMasksAliasesAndGuards) {
  std::mt19937 random(58210);
  for (bool rdna4 : {false, true})
    for (bool wide : {false, true})
      for (unsigned guest = 0; guest < 8; ++guest)
        for (unsigned mode = 0; mode < 512; ++mode) {
          uint32_t initial[4][66], words[4][66], expected[4][66];
          for (unsigned reg = 0; reg < 4; ++reg)
            for (unsigned lane = 0; lane < 66; ++lane)
              initial[reg][lane] =
                  lane % 2
                      ? random()
                      : goc_test::mullit_capture_values[(lane * (2 * reg + 1) + mode + reg) % 16];
          bool shared = mode & 1;
          unsigned target = (mode / 2) % 4;
          uint64_t descriptor = mode % 8 ? goc_test::dpp_modes[mode % 8 - 1] : 0;
          if (wide && (descriptor & GOC_DPP16))
            descriptor =
                (descriptor & ~GOC_DPP_ROW_MASK) | (uint64_t(mode & 15) << GOC_DPP_ROW_SHIFT);
          uint64_t exec_mask = mode & 2 ? 0x87654321fedcba98ULL : UINT64_MAX;
          if (!wide)
            exec_mask = uint32_t(exec_mask);
          std::memcpy(expected, initial, sizeof(expected));
          uint64_t writes = 0;
          for (unsigned lane = 0; lane < (wide ? 64U : 32U); ++lane) {
            int source = int(lane);
            bool active = descriptor ? goc_test::dpp_source(descriptor, exec_mask, lane, source)
                                     : ((exec_mask >> lane) & 1);
            if (active) {
              writes |= 1ULL << lane;
              expected[target][lane + 1] = goc_test::mullit_reference(
                  source < 0 ? 0 : initial[0][source + 1], initial[shared ? 0 : 1][lane + 1],
                  initial[shared ? 0 : 2][lane + 1], mode, !rdna4, policy(guest));
            }
          }
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
            std::memcpy(words, initial, sizeof(words));
            const uint32_t *a[] = {words[0] + 1}, *b[] = {words[shared ? 0 : 1] + 1},
                           *c[] = {words[shared ? 0 : 2] + 1};
            uint32_t *d[] = {words[target] + 1}, state = 0x12345678;
            ASSERT_EQ(call(rdna4, wide, cpu | policy(guest), exec_mask, descriptor | mode, d, a, b,
                           c, &state),
                      GOC_SUCCESS);
            ASSERT_EQ(state, 0x12345678U);
            for (unsigned reg = 0; reg < 4; ++reg)
              for (unsigned lane = 0; lane < 66; ++lane) {
                if (reg == target && lane > 0 && lane <= 64 && ((writes >> (lane - 1)) & 1)) {
                  ASSERT_TRUE(goc_test::mullit_result(words[reg][lane], expected[reg][lane]))
                      << rdna4 << "/" << wide << "/" << guest << "/" << mode << "/" << cpu << "/"
                      << reg << "/" << lane;
                } else {
                  ASSERT_EQ(words[reg][lane], expected[reg][lane])
                      << rdna4 << "/" << wide << "/" << guest << "/" << mode << "/" << cpu << "/"
                      << reg << "/" << lane;
                }
              }
          }
        }
}

TEST(MullitArchitecture, Rdna3LightingCartesianValues) {
  // rocjitsu ffc144c564c execute_v_mullit_f32_vop3 and finalize_omod_f32:
  // lighting sentinels, zero multiplication, and post-scaling +0 normalization.
  for (unsigned mode = 0; mode < 512; ++mode)
    for (unsigned start = 0; start < 4096; start += 64) {
      uint32_t words[4][64], want[64];
      for (unsigned lane = 0; lane < 64; ++lane) {
        unsigned index = start + lane;
        words[0][lane] = goc_test::mullit_capture_values[index / 256];
        words[1][lane] = goc_test::mullit_capture_values[index / 16 % 16];
        words[2][lane] = goc_test::mullit_capture_values[index % 16];
        want[lane] = goc_test::mullit_reference(words[0][lane], words[1][lane], words[2][lane],
                                                mode, true, policy(2));
      }
      const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
      uint32_t *d[] = {words[3]};
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
        ASSERT_EQ(call(false, true, cpu | policy(2), UINT64_MAX, mode, d, a, b, c), GOC_SUCCESS);
        for (unsigned lane = 0; lane < 64; ++lane)
          ASSERT_TRUE(goc_test::mullit_result(words[3][lane], want[lane]))
              << cpu << "/" << mode << "/" << start << "/" << lane;
      }
    }
}

TEST(MullitArchitecture, LiteralScalingAndDenormalDifferences) {
  // Tiny products can normalize on RDNA3, and negative normal products that
  // become tiny map to +0 rather than RDNA4's signed zero.
  struct Case {
    uint32_t a, b, c, mode;
    unsigned guest;
    uint32_t rdna3, rdna4;
  };

  const Case cases[] = {
      {0x3f800000, 0x3f800000, 0x3f800000, GOC_ALU_OMOD_2, 0, 0x3f800000, 0x40000000},
      {0x3f800000, 0x3f800000, 0x3f800000, GOC_ALU_OMOD_2, 6, 0x3f800000, 0x40000000},
      {0x00400000, 0x3f800000, 0x3f800000, GOC_ALU_OMOD_2, 2, 0x00800000, 0},
      {0x80800000, 0x3f800000, 0x3f800000, GOC_ALU_OMOD_HALF, 2, 0, 0x80000000},
      {0x3f800000, 0x3f800000, 1, 0, 1, 0xff7fffff, 0xff7fffff},
      {0x80400000, 0x3f800000, 0x3f800000, 0, 2, 0x80000000, 0x80000000}};
  for (const auto &v : cases)
    for (bool rdna4 : {false, true})
      for (bool wide : {false, true}) {
        uint32_t a0[64], b0[64], c0[64], output[64];
        std::fill_n(a0, 64, v.a);
        std::fill_n(b0, 64, v.b);
        std::fill_n(c0, 64, v.c);
        const uint32_t *a[] = {a0}, *b[] = {b0}, *c[] = {c0};
        uint32_t *d[] = {output};
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
          ASSERT_EQ(call(rdna4, wide, cpu | policy(v.guest), UINT64_MAX, v.mode, d, a, b, c),
                    GOC_SUCCESS);
          for (unsigned lane = 0; lane < (wide ? 64U : 32U); ++lane)
            EXPECT_EQ(output[lane], rdna4 ? v.rdna4 : v.rdna3);
        }
      }
}

TEST(MullitArchitecture, ReportingErrorsAndHostControls) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  uint32_t input[64], output[64];
  std::fill_n(input, 64, 0x3f803abcU);
  const uint32_t *a[] = {input};
  uint32_t *d[] = {output};
  for (bool rdna4 : {false, true})
    for (bool wide : {false, true}) {
      uint32_t state = 0x12345678;
      for (uint64_t mode : {uint64_t(0), goc_test::dpp_modes[0]}) {
        for (uint64_t semantics :
             {GOC_SEMANTICS_EXACT_EMPIRICAL, GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT})
          for (uint64_t exec_mask : {0ULL, ~0ULL})
            EXPECT_EQ(call(rdna4, wide, semantics, exec_mask, mode, nullptr, nullptr, nullptr,
                           nullptr, &state),
                      GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
        EXPECT_EQ(call(rdna4, wide, GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, mode,
                       nullptr, nullptr, nullptr, nullptr),
                  GOC_ERROR_UNSUPPORTED_SEMANTICS);
        EXPECT_EQ(call(rdna4, wide, 0, 0, mode, nullptr, nullptr, nullptr, nullptr, &state),
                  GOC_SUCCESS);
      }
      EXPECT_EQ(state, 0x12345678U);
      EXPECT_EQ(call(rdna4, wide, 0, 0, 1U << 9, nullptr, nullptr, nullptr, nullptr),
                GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(call(rdna4, wide, 1ULL << 63, 0, 0, nullptr, nullptr, nullptr, nullptr),
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
            ASSERT_EQ(
                call(rdna4, wide, cpu | policy(3), UINT64_MAX, GOC_ALU_OMOD_4, d, a, a, a, &state),
                GOC_SUCCESS);
            EXPECT_EQ(state, 0x12345678U);
            EXPECT_EQ(std::fegetround(), rounding);
#if defined(__x86_64__) || defined(_M_X64)
            EXPECT_EQ(_mm_getcsr() & ~0x3fU, controls);
#endif
          }
    }
}

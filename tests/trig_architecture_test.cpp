// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "dpp_reference.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "trig_golden.h"

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

const uint64_t exact = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;
using Fn32 = decltype(&goc_v_sin_f32);
using Fn64 = decltype(&goc_v_sin_f32_wave64);
const Fn32 functions[2][2] = {{goc_v_sin_f32, goc_v_cos_f32},
                              {goc_v_sin_f32_rdna4, goc_v_cos_f32_rdna4}};
const Fn64 wide_functions[2][2] = {{goc_v_sin_f32_wave64, goc_v_cos_f32_wave64},
                                   {goc_v_sin_f32_rdna4_wave64, goc_v_cos_f32_rdna4_wave64}};

uint64_t policy(unsigned guest) {
  return (guest & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
         (guest & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) | (guest & 4 ? GOC_FP_IEEE_MODE : 0);
}

int call(unsigned arch, bool wide, unsigned cosine, uint64_t flags, uint64_t exec_mask,
         uint64_t mode, uint32_t *const *d, const uint32_t *const *a, uint32_t *state = nullptr) {
  if (wide)
    return wide_functions[arch][cosine](flags, exec_mask, mode, d, a, state);
  return functions[arch][cosine](flags, uint32_t(exec_mask), mode, d, a, state);
}

float as_float(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

uint32_t as_bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

// Apply the borrowed architecture policies around existing captured results.
// Power-of-two scaling in double provides an independent rounding reference.
uint32_t reference(uint32_t raw, unsigned arch, unsigned cosine, unsigned guest, uint32_t mode) {
  if (mode & GOC_ALU_ABS_A)
    raw &= 0x7fffffff;
  if (mode & GOC_ALU_NEG_A)
    raw ^= 0x80000000;
  uint32_t magnitude = raw & 0x7fffffff, value = 0;
  if (magnitude > 0x7f800000)
    value = raw | (arch || (guest & 4) ? 0x400000 : 0);
  else if (magnitude == 0x7f800000)
    value = 0xffc00000;
  else if (!magnitude || ((guest & 1) && magnitude < 0x800000))
    value = cosine ? 0x3f800000 : raw & 0x80000000;
  else {
    bool found = false;
    for (const auto &row : goc_test::trig_golden)
      if ((row[0] & 0x7fffffff) == magnitude) {
        value = row[cosine + 1];
        if (!cosine && ((value & 0x7fffffff) || magnitude < 0x39c00000))
          value ^= (raw ^ row[0]) & 0x80000000;
        found = true;
        break;
      }
    EXPECT_TRUE(found);
  }
  if ((guest & 2) && (value & 0x7fffffff) < 0x800000)
    value &= 0x80000000;
  unsigned omod = !arch && (guest & 4) ? 0 : ((mode >> 6) & 3);
  if (omod && (value & 0x7fffffff) < 0x7f800000) {
    if (!arch && (value & 0x7fffffff) < 0x800000)
      value = 0;
    else if (!arch && omod == 3 && (value & 0x7fffffff) < 0x1000000)
      value &= 0x80000000;
    else
      value = as_bits(float(std::ldexp(double(as_float(value)), omod == 3 ? -1 : int(omod))));
  }
  if (mode & GOC_ALU_CLAMP) {
    if ((value >> 31) || (value & 0x7fffffff) > 0x7f800000)
      value = 0;
    else if (value > 0x3f800000)
      value = 0x3f800000;
  }
  if (arch && omod && (value & 0x7fffffff) < 0x800000)
    value = 0;
  return value;
}

} // namespace

TEST(TrigArchitecture, CapturedResultsGuestPoliciesModifiersDppMasksAndAliases) {
  for (unsigned arch = 0; arch < 2; ++arch)
    for (unsigned cosine = 0; cosine < 2; ++cosine)
      for (bool wide : {false, true})
        for (unsigned guest = 0; guest < 8; ++guest)
          for (unsigned variant = 0; variant < 32; ++variant) {
            unsigned mode = (variant & 1 ? GOC_ALU_ABS_A : 0) | (variant & 2 ? GOC_ALU_NEG_A : 0) |
                            ((variant >> 2 & 3) << 6) | (variant & 16 ? GOC_ALU_CLAMP : 0);
            for (unsigned permutation = 0; permutation < 8; ++permutation) {
              uint64_t descriptor = permutation ? goc_test::dpp_modes[permutation - 1] : 0;
              if (wide && (descriptor & GOC_DPP16))
                descriptor = (descriptor & ~GOC_DPP_ROW_MASK) |
                             (uint64_t(variant & 15) << GOC_DPP_ROW_SHIFT);
              uint64_t exec_mask = variant & 1 ? 0x87654321fedcba98ULL : UINT64_MAX;
              if (!wide)
                exec_mask = uint32_t(exec_mask);
              uint32_t initial[2][66], words[2][66], expected[2][66];
              for (unsigned reg = 0; reg < 2; ++reg)
                std::fill_n(initial[reg], 66, 0x12345678U);
              for (unsigned lane = 0; lane < 64; ++lane)
                initial[0][lane + 1] =
                    lane % 8 == 7 ? 0x7f812345
                                  : goc_test::trig_golden[(lane + variant) %
                                                          (sizeof(goc_test::trig_golden) /
                                                           sizeof(goc_test::trig_golden[0]))][0];
              unsigned target = permutation & 1;
              std::memcpy(expected, initial, sizeof(expected));
              uint64_t writes = 0;
              for (unsigned lane = 0; lane < (wide ? 64U : 32U); ++lane) {
                int source = int(lane);
                bool active = descriptor ? goc_test::dpp_source(descriptor, exec_mask, lane, source)
                                         : ((exec_mask >> lane) & 1);
                if (active) {
                  writes |= 1ULL << lane;
                  expected[target][lane + 1] =
                      reference(source < 0 ? 0 : initial[0][source + 1], arch, cosine, guest, mode);
                }
              }
              for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
                for (bool loose : {false, true}) {
                  std::memcpy(words, initial, sizeof(words));
                  const uint32_t *a[] = {words[0] + 1};
                  uint32_t *d[] = {words[target] + 1}, state = 0x12345678;
                  ASSERT_EQ(call(arch, wide, cosine, cpu | policy(guest) | (loose ? 0 : exact),
                                 exec_mask, descriptor | mode, d, a, loose ? &state : nullptr),
                            GOC_SUCCESS);
                  ASSERT_EQ(state, 0x12345678U);
                  for (unsigned reg = 0; reg < 2; ++reg)
                    for (unsigned lane = 0; lane < 66; ++lane) {
                      uint32_t got = words[reg][lane], want = expected[reg][lane];
                      bool written =
                          reg == target && lane > 0 && lane < 65 && ((writes >> (lane - 1)) & 1);
                      unsigned magnitude = want & 0x7fffffff;
                      if (loose && written && got != want && magnitude && magnitude < 0x7f800000) {
                        double tolerance = magnitude < 0x800000 ? 3 * 0x1p-149 : 8e-7;
                        ASSERT_NEAR(as_float(got), as_float(want), tolerance)
                            << arch << "/" << cosine << "/" << guest << "/" << variant;
                      } else
                        ASSERT_EQ(got, want) << arch << "/" << cosine << "/" << guest << "/"
                                             << variant << "/" << lane;
                    }
                }
            }
          }
}

TEST(TrigArchitecture, LiteralNaNFlushingAndIeeePolicy) {
  struct Case {
    uint32_t input, mode;
    unsigned guest;
    uint32_t rdna3, rdna4;
  };

  const Case cases[] = {{0x7f812345, 0, 0, 0x7f812345, 0x7fc12345},
                        {0xff812345, 0, 4, 0xffc12345, 0xffc12345},
                        {0x3e800000, GOC_ALU_OMOD_2, 4, 0x3f800000, 0x40000000},
                        {0x3e800000, GOC_ALU_OMOD_2, 0, 0x40000000, 0x40000000},
                        {0x80000001, 0, 1, 0x80000000, 0x80000000},
                        {0x80000001, 0, 2, 0x80000000, 0x80000000},
                        {0x00080000, GOC_ALU_OMOD_4, 0, 0, 0x00c90fd4},
                        {0x80200000, GOC_ALU_OMOD_HALF, 0, 0x80000000, 0}};
  for (const auto &v : cases)
    for (unsigned arch = 0; arch < 2; ++arch)
      for (bool wide : {false, true}) {
        uint32_t input[64], output[64];
        std::fill_n(input, 64, v.input);
        const uint32_t *a[] = {input};
        uint32_t *d[] = {output};
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
          for (bool loose : {false, true}) {
            ASSERT_EQ(call(arch, wide, 0, cpu | policy(v.guest) | (loose ? 0 : exact), UINT64_MAX,
                           v.mode, d, a),
                      GOC_SUCCESS);
            for (unsigned lane = 0; lane < (wide ? 64U : 32U); ++lane)
              EXPECT_EQ(output[lane], arch ? v.rdna4 : v.rdna3);
          }
      }
}

TEST(TrigArchitecture, ReportingErrorsAndHostControls) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  uint32_t input[64], output[64];
  for (unsigned lane = 0; lane < 64; ++lane)
    input[lane] = lane & 1 ? 0x7f812345 : 0x80000001 + lane;
  const uint32_t *a[] = {input};
  uint32_t *d[] = {output};
  for (unsigned arch = 0; arch < 2; ++arch)
    for (bool wide : {false, true})
      for (unsigned cosine = 0; cosine < 2; ++cosine) {
        uint32_t state = 0x80000000;
        for (unsigned guest = 0; guest < 8; ++guest)
          for (uint64_t semantics : {GOC_SEMANTICS_EXACT_EMPIRICAL, exact})
            if (!arch || (guest & 3)) {
              for (uint64_t exec_mask : {0ULL, ~0ULL})
                EXPECT_EQ(call(arch, wide, cosine, semantics | policy(guest), exec_mask, 0, nullptr,
                               nullptr, &state),
                          GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
            }
        EXPECT_EQ(state, 0x80000000U);
        EXPECT_EQ(call(arch, wide, cosine, 0, 0, GOC_ALU_NEG_B, nullptr, nullptr),
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
              ASSERT_EQ(call(arch, wide, cosine, cpu | exact | policy(3), UINT64_MAX,
                             GOC_ALU_OMOD_4, d, a),
                        GOC_SUCCESS);
              EXPECT_EQ(std::fegetround(), rounding);
#if defined(__x86_64__) || defined(_M_X64)
              EXPECT_EQ(_mm_getcsr() & ~0x3fU, controls);
#endif
            }
      }
}

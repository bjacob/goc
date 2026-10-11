// SPDX-License-Identifier: MIT

#include "dpp_reference.h"
#include "fp_environment.h"
#include "fuzzy.h"
#include "goc/goc.h"
#include "half_reference.h"
#include "loose_tolerances.h"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace {

const uint64_t exact = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;
using Fn32 = decltype(&goc_v_sin_f16);
using Fn64 = decltype(&goc_v_sin_f16_wave64);
const Fn32 functions[2][2] = {{goc_v_sin_f16, goc_v_cos_f16},
                              {goc_v_sin_f16_rdna4, goc_v_cos_f16_rdna4}};
const Fn64 wave_functions[2][2] = {{goc_v_sin_f16_wave64, goc_v_cos_f16_wave64},
                                   {goc_v_sin_f16_rdna4_wave64, goc_v_cos_f16_rdna4_wave64}};
const decltype(&goc_v_sin_f32) wide_functions[] = {goc_v_sin_f32_rdna4, goc_v_cos_f32_rdna4};

uint64_t policy(unsigned guest) {
  return (guest & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
         (guest & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) | (guest & 4 ? GOC_FP_IEEE_MODE : 0);
}

int call(unsigned arch, bool wide, unsigned cosine, uint64_t flags, uint64_t exec_mask,
         uint64_t mode, uint32_t *const *d, const uint32_t *const *a, uint32_t *state = nullptr) {
  return wide ? wave_functions[arch][cosine](flags, exec_mask, mode, d, a, state)
              : functions[arch][cosine](flags, uint32_t(exec_mask), mode, d, a, state);
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

// This is a differential integration reference, not an independent numerical oracle.
// Round the independently tested FP32 exact API's results using the numeric
// half reference, preserving NaN signs/payloads explicitly.
std::vector<uint16_t> reference_table(int op) {
  std::vector<uint16_t> table(65536);
  for (unsigned start = 0; start < 65536; start += 32) {
    uint32_t a[32], d[32];
    const uint32_t *ap[] = {a};
    uint32_t *dp[] = {d};
    for (unsigned lane = 0; lane < 32; ++lane) {
      uint16_t half = uint16_t(start + lane);
      a[lane] = (half & 0x7fff) > 0x7c00
                    ? (uint32_t(half & 0x8000) << 16) | 0x7f800000 | (uint32_t(half & 1023) << 13)
                    : as_bits(float(goc_test::half_value(half)));
    }
    EXPECT_EQ(wide_functions[op](GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, UINT32_MAX,
                                 0, dp, ap, nullptr),
              GOC_SUCCESS);
    for (unsigned lane = 0; lane < 32; ++lane)
      table[start + lane] =
          (d[lane] & 0x7fffffff) > 0x7f800000
              ? uint16_t(((d[lane] >> 16) & 0x8000) | 0x7c00 | ((d[lane] >> 13) & 1023))
              : goc_test::half_bits(double(as_float(d[lane])));
  }
  return table;
}

uint16_t scale_result(uint16_t value, int mode) {
  double result = goc_test::half_value(value);
  const double scales[] = {1, 2, 4, 0.5};
  if (mode & 12) {
    if (std::abs(result) < 0x1p-14)
      result = 0;
    else
      result *= scales[(mode >> 2) & 3];
    if (!std::isnan(result)) {
      value = goc_test::half_bits(result);
      if ((value & 0x7fff) < 0x400)
        value = 0;
    }
  }
  if (mode & 16) {
    if (std::isnan(result) || (value & 0x8000))
      return 0;
    if (value > 0x3c00)
      return 0x3c00;
  }
  return value;
}

uint16_t reference(uint32_t raw, const std::vector<uint16_t> &table, unsigned arch, unsigned guest,
                   uint32_t mode) {
  uint16_t input = uint16_t(raw >> (mode & GOC_ALU_HIGH_A ? 16 : 0));
  if (mode & GOC_ALU_ABS_A)
    input &= 0x7fff;
  if (mode & GOC_ALU_NEG_A)
    input ^= 0x8000;
  if ((guest & 1) && (input & 0x7fff) < 0x400)
    input &= 0x8000;
  uint16_t value = table[input];
  if (!arch && !(guest & 4) && (input & 0x7fff) > 0x7c00)
    value = input;
  if ((guest & 2) && (value & 0x7fff) < 0x400)
    value &= 0x8000;
  unsigned omod = !arch && (!(guest & 2) || (guest & 4)) ? 0 : ((mode >> 6) & 3);
  return scale_result(value, (omod << 2) | (mode & GOC_ALU_CLAMP ? 16 : 0));
}

} // namespace

TEST(HalfTrigArchitecture, EveryEncodingGuestPolicyAndOutputModifier) {
  for (unsigned cosine = 0; cosine < 2; ++cosine) {
    auto table = reference_table(cosine);
    for (unsigned arch = 0; arch < 2; ++arch)
      for (unsigned guest = 0; guest < 8; ++guest)
        for (unsigned omod = 0; omod < 4; ++omod)
          for (unsigned start = 0; start < 65536; start += 64) {
            uint32_t input[64], output[64];
            const uint32_t *a[] = {input};
            uint32_t *d[] = {output};
            for (unsigned lane = 0; lane < 64; ++lane)
              input[lane] = start + lane;
            for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
              std::fill_n(output, 64, 0xabcd0000U);
              ASSERT_EQ(call(arch, true, cosine, cpu | exact | policy(guest), UINT64_MAX, omod << 6,
                             d, a),
                        GOC_SUCCESS);
              for (unsigned lane = 0; lane < 64; ++lane)
                ASSERT_EQ(output[lane],
                          0xabcd0000U | reference(input[lane], table, arch, guest, omod << 6))
                    << arch << "/" << cosine << "/" << guest << "/" << omod << "/" << input[lane];
            }
          }
  }
}

TEST(HalfTrigArchitecture, ModifiersDppMasksHalfSelectionAndAliases) {
  const uint16_t inputs[] = {0,      0x8000, 1,      0x8001, 0x3ff,  0x8400, 0x1000, 0x3400,
                             0xb400, 0x3c00, 0x6400, 0x7bff, 0x7c00, 0xfc00, 0x7c01, 0xfe23};
  for (unsigned cosine = 0; cosine < 2; ++cosine) {
    auto table = reference_table(cosine);
    for (unsigned arch = 0; arch < 2; ++arch)
      for (bool wide : {false, true})
        for (unsigned guest = 0; guest < 8; ++guest)
          for (unsigned variant = 0; variant < 128; ++variant) {
            unsigned mode = (variant & 1 ? GOC_ALU_ABS_A : 0) | (variant & 2 ? GOC_ALU_NEG_A : 0) |
                            ((variant >> 2 & 3) << 6) | (variant & 16 ? GOC_ALU_CLAMP : 0) |
                            (variant & 32 ? GOC_ALU_HIGH_A : 0) |
                            (variant & 64 ? GOC_ALU_HIGH_D : 0);
            for (unsigned permutation = 0; permutation < 8; ++permutation) {
              uint64_t descriptor = permutation ? goc_test::dpp_modes[permutation - 1] : 0;
              if (wide && (descriptor & GOC_DPP16))
                descriptor = (descriptor & ~GOC_DPP_ROW_MASK) |
                             (uint64_t(variant & 15) << GOC_DPP_ROW_SHIFT);
              uint64_t exec_mask = variant & 1 ? 0x87654321fedcba98ULL : UINT64_MAX;
              if (!wide)
                exec_mask = uint32_t(exec_mask);
              uint32_t initial[2][66], words[2][66], expected[2][66];
              for (auto &reg : initial)
                std::fill_n(reg, 66, 0x12345678U);
              for (unsigned lane = 0; lane < 64; ++lane)
                initial[0][lane + 1] = inputs[(lane + variant) % 16] |
                                       (uint32_t(inputs[(lane + variant + 7) % 16]) << 16);
              unsigned target = permutation & 1, shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
              std::memcpy(expected, initial, sizeof(expected));
              uint64_t writes = 0;
              for (unsigned lane = 0; lane < (wide ? 64U : 32U); ++lane) {
                int source = int(lane);
                bool active = descriptor ? goc_test::dpp_source(descriptor, exec_mask, lane, source)
                                         : ((exec_mask >> lane) & 1);
                if (active) {
                  writes |= 1ULL << lane;
                  uint16_t result =
                      reference(source < 0 ? 0 : initial[0][source + 1], table, arch, guest, mode);
                  expected[target][lane + 1] = (expected[target][lane + 1] & ~(0xffffU << shift)) |
                                               (uint32_t(result) << shift);
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
                  if (loose) {
                    ASSERT_EQ(state, 0x12345678U);
                  }
                  for (unsigned reg = 0; reg < 2; ++reg)
                    for (unsigned lane = 0; lane < 66; ++lane) {
                      uint32_t got = words[reg][lane], want = expected[reg][lane];
                      bool written =
                          reg == target && lane > 0 && lane < 65 && ((writes >> (lane - 1)) & 1);
                      unsigned magnitude = (want >> shift) & 0x7fff;
                      if (loose && written && magnitude && magnitude < 0x7c00) {
                        ASSERT_EQ(got & ~(0x7fffU << shift), want & ~(0x7fffU << shift));
                        ASSERT_TRUE(goc_test::fuzzy(goc_test::half_value(uint16_t(got >> shift)),
                                                    goc_test::half_value(uint16_t(want >> shift)),
                                                    goc_test::half_sfu_tolerance))
                            << arch << "/" << cosine << "/" << guest << "/" << variant << "/"
                            << permutation << "/" << cpu << "/" << lane;
                      } else {
                        ASSERT_EQ(got, want) << arch << "/" << cosine << "/" << guest << "/"
                                             << variant << "/" << lane;
                      }
                    }
                }
            }
          }
  }
}

TEST(HalfTrigArchitecture, LiteralNaNFlushingAndIeeePolicy) {
  struct Case {
    uint16_t input;
    uint32_t mode;
    unsigned guest;
    uint16_t rdna3, rdna4;
  };

  const Case cases[] = {{0x7c01, 0, 0, 0x7c01, 0x7e01},
                        {0xfc01, 0, 4, 0xfe01, 0xfe01},
                        {0x3400, GOC_ALU_OMOD_2, 0, 0x3c00, 0x4000},
                        {0x3400, GOC_ALU_OMOD_2, 2, 0x4000, 0x4000},
                        {0x3400, GOC_ALU_OMOD_2, 6, 0x3c00, 0x4000},
                        {0x8001, 0, 1, 0x8000, 0x8000},
                        {0x8001, 0, 2, 0x8000, 0x8000},
                        {0x8001, GOC_ALU_OMOD_4, 0, 0x8006, 0}};
  for (const auto &v : cases)
    for (unsigned arch = 0; arch < 2; ++arch)
      for (bool wide : {false, true}) {
        uint32_t input[64], output[64];
        std::fill_n(input, 64, v.input);
        const uint32_t *a[] = {input};
        uint32_t *d[] = {output};
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
          for (bool loose : {false, true}) {
            std::fill_n(output, 64, 0xabcd0000U);
            ASSERT_EQ(call(arch, wide, 0, cpu | policy(v.guest) | (loose ? 0 : exact), UINT64_MAX,
                           v.mode, d, a),
                      GOC_SUCCESS);
            for (unsigned lane = 0; lane < (wide ? 64U : 32U); ++lane)
              EXPECT_EQ(output[lane], 0xabcd0000U | (arch ? v.rdna4 : v.rdna3));
          }
      }
}

TEST(HalfTrigArchitecture, ReportingErrorsAndHostControls) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  uint32_t input[64], output[64];
  for (unsigned lane = 0; lane < 64; ++lane)
    input[lane] = lane & 1 ? 0x7c01 : 0x8001 + lane;
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

// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_exec_masks.h"
#include "rdna4_half_fma_reference.h"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <random>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_rdna4_v_fmac_f32);
const Fn functions[] = {goc_rdna4_v_fmac_f32, goc_rdna4_v_fmac_f16};
const uint32_t known32 = GOC_ALU_NEG_A | GOC_ALU_NEG_B | GOC_ALU_ABS_A | GOC_ALU_ABS_B |
                         GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP;
const uint32_t known16 = known32 | GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_D;
const uint32_t special32[] = {0,          0x80000000, 1,          0x80000001, 0x007fffff,
                              0x00800000, 0x3f800000, 0x3f800001, 0x3f7fffff, 0x3f000000,
                              0xbf000000, 0x40000000, 0xc0000000, 0xbf800000, 0x7f7fffff,
                              0xff7fffff, 0x7f800000, 0xff800000, 0x7f800001, 0xffc01234};
const uint16_t special16[] = {0,      0x8000, 1,      0x8001, 0x3ff,  0x400,  0x3c00,
                              0x3c01, 0x3bff, 0x3800, 0xb800, 0x4000, 0xc000, 0xbc00,
                              0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7c01, 0xfe12};

float as_float(uint32_t bits) {
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

uint32_t as_bits(float value) {
  uint32_t result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

uint32_t reference(bool half, uint32_t a, uint32_t b, uint32_t d, uint32_t mode, bool saturate) {
  if (half) {
    uint32_t full_mode = mode | ((mode & GOC_ALU_HIGH_D) ? GOC_ALU_HIGH_C : 0);
    uint16_t result = goc_test::half_fma_reference::evaluate(a, b, d, full_mode, saturate);
    return mode & GOC_ALU_HIGH_D ? (d & 0xffff) | (uint32_t(result) << 16)
                                 : (d & 0xffff0000) | result;
  }
  if (mode & GOC_ALU_ABS_A)
    a &= 0x7fffffff;
  if (mode & GOC_ALU_NEG_A)
    a ^= 0x80000000;
  if (mode & GOC_ALU_ABS_B)
    b &= 0x7fffffff;
  if (mode & GOC_ALU_NEG_B)
    b ^= 0x80000000;
  float result = std::fma(as_float(a), as_float(b), as_float(d));
  const float scale[] = {1, 2, 4, 0.5f};
  result *= scale[(mode >> 6) & 3];
  if (mode & GOC_ALU_CLAMP)
    result = !(result > 0) ? 0 : std::min(result, 1.0f);
  return as_bits(result);
}

void check(bool half, uint32_t actual, uint32_t want, uint32_t mode, bool exact) {
  uint32_t mask = half ? ((mode & GOC_ALU_HIGH_D) ? 0xffff0000 : 0xffff) : UINT32_MAX;
  EXPECT_EQ(actual & ~mask, want & ~mask);
  uint32_t shift = half && (mode & GOC_ALU_HIGH_D) ? 16 : 0;
  uint32_t got = (actual & mask) >> shift, expected = (want & mask) >> shift;
  uint32_t magnitude = half ? 0x7fff : 0x7fffffff, infinity = half ? 0x7c00 : 0x7f800000;
  if (!exact && (expected & magnitude) > infinity) {
    EXPECT_GT(got & magnitude, infinity);
  } else {
    EXPECT_EQ(got, expected);
  }
}

void fill(bool half, uint32_t (&words)[3][34], unsigned seed) {
  std::mt19937 random(seed);
  for (int reg = 0; reg < 3; ++reg) {
    std::fill(words[reg], words[reg] + 34, 0xfacecafe);
    for (int lane = 1; lane <= 32; ++lane) {
      int index = (lane + reg * 7) % 20;
      words[reg][lane] = lane > 20 ? random()
                         : half ? special16[index] | (uint32_t(special16[(index + 7) % 20]) << 16)
                                : special32[index];
    }
  }
}

void run(bool half, uint64_t flags, uint64_t mask, uint64_t mode, int a, int b, int d,
         uint32_t (&words)[3][34]) {
  uint32_t before[3][34];
  std::memcpy(before, words, sizeof(before));
  uint32_t *p[] = {words[0] + 1, words[1] + 1, words[2] + 1};
  ASSERT_EQ(functions[half](flags, mask, mode, p + d, p + a, p + b), GOC_SUCCESS);
  for (int reg = 0; reg < 3; ++reg)
    for (int lane = 0; lane < 34; ++lane) {
      if (reg == d && lane >= 1 && lane <= 32 && ((mask >> (lane - 1)) & 1)) {
        check(half, words[reg][lane],
              reference(half, before[a][lane], before[b][lane], before[d][lane], mode,
                        flags & GOC_FP16_OVFL),
              mode, half && (flags & GOC_SEMANTICS_MASK));
      } else {
        EXPECT_EQ(words[reg][lane], before[reg][lane]);
      }
    }
}

} // namespace

TEST(Fmac, AllModifiersOverflowPoliciesAndCpuLevels) {
  for (bool half : {false, true})
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool exact : {false, true})
        for (bool saturate : {false, true})
          for (uint32_t mode = 0; mode < (half ? 8192u : 512u); ++mode) {
            if (mode & ~(half ? known16 : known32))
              continue;
            SCOPED_TRACE(::testing::Message()
                         << half << '/' << cpu << '/' << exact << '/' << saturate << '/' << mode);
            uint64_t flags = cpu | (saturate ? GOC_FP16_OVFL : 0) |
                             (exact ? GOC_SEMANTICS_EXACT_EMPIRICAL : 0) |
                             (exact && half ? GOC_SEMANTICS_STRICT : 0);
            uint32_t words[3][34];
            fill(half, words, mode + 198);
            run(half, flags, UINT32_MAX, mode, 0, 1, mode % 3, words);
          }
}

TEST(Fmac, MasksAndAllAccumulatorAliases) {
  const uint32_t modes[] = {0,
                            GOC_ALU_NEG_A,
                            GOC_ALU_ABS_A | GOC_ALU_NEG_B,
                            GOC_ALU_OMOD_2,
                            GOC_ALU_OMOD_4 | GOC_ALU_CLAMP,
                            GOC_ALU_ABS_B | GOC_ALU_OMOD_HALF};
  for (bool half : {false, true})
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool exact : {false, true})
        for (auto base_mode : modes)
          for (uint32_t selectors = 0; selectors < (half ? 8u : 1u); ++selectors)
            for (auto mask : rdna4_exec_masks())
              for (int b : {0, 1})
                for (int d = 0; d < 3; ++d) {
                  uint32_t mode =
                      base_mode | ((selectors & 3) << 9) | ((selectors & 4) ? GOC_ALU_HIGH_D : 0);
                  SCOPED_TRACE(::testing::Message() << half << '/' << cpu << '/' << mode << '/'
                                                    << mask << '/' << b << '/' << d);
                  uint32_t words[3][34];
                  fill(half, words, 429);
                  run(half, cpu | (exact ? GOC_SEMANTICS_EXACT_EMPIRICAL : 0), mask, mode, 0, b, d,
                      words);
                }
}

TEST(Fmac, HalfEveryEncodingAndFusedRounding) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (bool exact : {false, true}) {
      std::mt19937 random(45239);
      for (unsigned base = 0; base < 65536; base += 32) {
        SCOPED_TRACE(::testing::Message() << cpu << '/' << exact << '/' << base);
        uint32_t words[3][34];
        fill(true, words, base);
        for (int lane = 1; lane <= 32; ++lane) {
          words[0][lane] = (base + lane - 1) * 0x10001u;
          words[1][lane] = random();
          words[2][lane] = random();
        }
        uint32_t mode = ((base / 32) % 4) << 6;
        if (base & 32)
          mode |= GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_D;
        run(true, cpu | (exact ? GOC_SEMANTICS_EXACT_EMPIRICAL : 0), UINT32_MAX, mode, 0, 1, 2,
            words);
      }
    }
  // The product must remain fused until the accumulator is added.
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint32_t words[3][32];
    uint32_t *p[] = {words[0], words[1], words[2]};
    std::fill(words[0], words[0] + 32, 0x3f800001);
    std::fill(words[1], words[1] + 32, 0x3f7ffffe);
    std::fill(words[2], words[2] + 32, 0xbf800000);
    ASSERT_EQ(goc_rdna4_v_fmac_f32(cpu, UINT32_MAX, 0, p + 2, p, p + 1), GOC_SUCCESS);
    for (auto word : words[2])
      EXPECT_EQ(word, 0xa8800000u);
  }
}

TEST(Fmac, HalfHardwareWitnessesAndHostEnvironment) {
  // rocjitsu's f16_fma_omod_cases has matching FMA/FMAC gfx1201 captures.
  struct Case {
    uint16_t a, b, d, expected;
  };

  const Case cases[] = {{1, 0x3400, 0x8400, 0x8000},
                        {0x400, 0x400, 0x7bff, 0x77ff},
                        {0, 0x7c00, 0x7e01, 0xfe00},
                        {0x7c01, 0x3c00, 0, 0x7e01},
                        {0x3c01, 0x3e00, 0x8001, 0x3a01}};
  std::fenv_t saved;
  std::fegetenv(&saved);
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO})
      for (bool high : {false, true})
        for (auto test : cases) {
          std::fesetround(rounding);
          std::feclearexcept(FE_ALL_EXCEPT);
          std::feraiseexcept(FE_DIVBYZERO);
          uint32_t words[3][32];
          uint32_t *p[] = {words[0], words[1], words[2]};
          std::fill(words[0], words[0] + 32, uint32_t(test.a) * 0x10001);
          std::fill(words[1], words[1] + 32, uint32_t(test.b) * 0x10001);
          std::fill(words[2], words[2] + 32,
                    high ? (uint32_t(test.d) << 16) | 0x7c01 : 0xfc010000 | test.d);
          uint32_t mode =
              GOC_ALU_OMOD_HALF | (high ? GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_D : 0);
          EXPECT_EQ(goc_rdna4_v_fmac_f16(cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                                         UINT32_MAX, mode, p + 2, p, p + 1),
                    GOC_SUCCESS);
          for (auto word : words[2])
            EXPECT_EQ(word,
                      high ? (uint32_t(test.expected) << 16) | 0x7c01 : 0xfc010000 | test.expected);
          EXPECT_EQ(std::fegetround(), rounding);
          EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
        }
  std::fesetenv(&saved);
}

TEST(Fmac, ValidationAndZeroMasks) {
  for (bool half : {false, true}) {
    auto fn = functions[half];
    uint32_t words[32];
    std::fill(words, words + 32, 0xfacecafe);
    auto p = words;
    for (uint64_t mask : {UINT64_C(0), UINT64_C(0xffffffff00000000), UINT64_MAX}) {
      for (unsigned bit = 0; bit < 32; ++bit)
        if ((UINT32_C(1) << bit) & ~(half ? known16 : known32)) {
          EXPECT_EQ(fn(0, mask, UINT32_C(1) << bit, &p, &p, &p), GOC_ERROR_INVALID_FLAGS);
        }
      EXPECT_EQ(fn(UINT64_C(1) << 63, mask, 0, &p, &p, &p), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn((UINT64_C(2) << 16) | GOC_SEMANTICS_STRICT, mask, 0, &p, &p, &p),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
      if (!half) {
        EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, mask, 0, &p, &p, &p),
                  GOC_ERROR_UNSUPPORTED_SEMANTICS);
      }
    }
    for (auto word : words)
      EXPECT_EQ(word, 0xfacecafe);
    EXPECT_EQ(fn(0, 0, 0, nullptr, nullptr, nullptr), GOC_SUCCESS);
    EXPECT_EQ(fn(0, UINT64_C(0xffffffff00000000), 0, nullptr, nullptr, nullptr), GOC_SUCCESS);
    EXPECT_EQ(fn(UINT64_C(2) << 16, UINT32_MAX, 0, &p, &p, &p), GOC_SUCCESS);
  }
}

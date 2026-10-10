// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "dpp_reference.h"
#include "fma32_exact_hardware.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "mixed_fma_reference.h"
#include "omod_reference.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace {

const uint32_t edge[] = {0,          0x80000000, 1,          0x80000001, 0x7fffff,   0x800000,
                         0x3f800000, 0xbf800000, 0x40000000, 0x7f7fffff, 0x7effffff, 0x7f800000,
                         0xff800000, 0x7f800001, 0xffc00003, 0x1000000};

int invoke(bool rdna3, bool wide, bool accumulate, uint64_t flags, uint64_t exec_mask,
           uint64_t mode, uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c,
           uint32_t *state = nullptr) {
  if (accumulate) {
    auto f32 = rdna3 ? goc_v_fmac_f32 : goc_v_fmac_f32_rdna4;
    auto f64 = rdna3 ? goc_v_fmac_f32_wave64 : goc_v_fmac_f32_rdna4_wave64;
    return wide ? f64(flags, exec_mask, mode, &d, &a, &b, state)
                : f32(flags, uint32_t(exec_mask), mode, &d, &a, &b, state);
  }
  auto f32 = rdna3 ? goc_v_fma_f32 : goc_v_fma_f32_rdna4;
  auto f64 = rdna3 ? goc_v_fma_f32_wave64 : goc_v_fma_f32_rdna4_wave64;
  return wide ? f64(flags, exec_mask, mode, &d, &a, &b, &c, state)
              : f32(flags, uint32_t(exec_mask), mode, &d, &a, &b, &c, state);
}

uint64_t fp_flags(unsigned policy) {
  return (policy & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
         (policy & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) | (policy & 4 ? GOC_FP_IEEE_MODE : 0);
}

uint32_t reference(uint32_t a, uint32_t b, uint32_t c, uint32_t mode, uint64_t flags, bool rdna3) {
  if (rdna3 && (!(flags & GOC_FP_FLUSH_OUTPUT_DENORMALS) || (flags & GOC_FP_IEEE_MODE)))
    mode &= ~GOC_ALU_OMOD_HALF;
  uint64_t arithmetic_flags =
      flags | (mode & GOC_ALU_OMOD_HALF ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0);
  uint32_t result = goc_test::mixed_fma_reference::evaluate(0, a, b, c, 0, mode & 63, false, false,
                                                            arithmetic_flags, rdna3);
  // Host scaling is an independent oracle here; callers evaluate it under RNE
  // with denormals preserved, before varying the host environment.
  if ((result & 0x7fffffff) <= 0x7f800000) {
    float value;
    std::memcpy(&value, &result, sizeof(value));
    value = goc_test::omod_f32_reference(value, mode);
    std::memcpy(&result, &value, sizeof(result));
  }
  if (mode & GOC_ALU_CLAMP)
    result = (result & 0x80000000) || (result & 0x7fffffff) > 0x7f800000 ? 0
             : result > 0x3f800000                                       ? 0x3f800000
                                                                         : result;
  return result;
}

} // namespace

TEST(FmaExact, HardwareCorpora) {
  for (bool accumulate : {false, true})
    for (unsigned boundary = 0; boundary < 2; ++boundary)
      for (unsigned policy = 0; policy < 4; ++policy)
        for (bool wide : {false, true})
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
            uint64_t hash = goc_test::capture_hash_seed;
            unsigned width = wide ? 64 : 32;
            for (unsigned variant = 0; variant < 16; ++variant) {
              uint32_t mode = ((variant & 3) << 6) | (variant & 4 ? GOC_ALU_CLAMP : 0) |
                              (variant & 8 ? GOC_ALU_NEG_A | GOC_ALU_ABS_A | GOC_ALU_ABS_B |
                                                 (accumulate ? 0 : GOC_ALU_NEG_C)
                                           : 0);
              uint64_t flags =
                  cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT | fp_flags(policy);
              for (unsigned base = 0; base < 8192; base += width) {
                uint32_t words[4][64];
                for (unsigned lane = 0; lane < width; ++lane) {
                  unsigned i = base + lane;
                  if (boundary) {
                    unsigned index = i & 4095;
                    const uint32_t addends[2][8] = {
                        {0, 1, 2, 3, 0x80000000, 0x80000001, 0x80000002, 0x80000003},
                        {0, 1, 0x3f800000, 0xbf800000, 0x72800000, 0xf2800000, 0x73000000,
                         0xf3000000}};
                    words[0][lane] =
                        i < 4096 ? 0x800000 - 64 + (index >> 5) : 0x7f800000 - 1 - (index >> 5);
                    words[1][lane] = 0x3f800000 - 2 + ((index >> 3) & 3);
                    words[2][lane] = addends[i / 4096][index & 7];
                  } else {
                    uint64_t random = uint64_t(i) * 0x9e3779b97f4a7c15ULL;
                    for (unsigned operand = 0; operand < 3; ++operand) {
                      random ^= random >> 12;
                      random ^= random << 25;
                      random ^= random >> 27;
                      words[operand][lane] = i < 4096 ? edge[(i >> (8 - operand * 4)) & 15]
                                                      : uint32_t(random * 0x2545f4914f6cdd1dULL);
                    }
                  }
                  words[3][lane] = words[2][lane];
                }
                ASSERT_EQ(invoke(false, wide, accumulate, flags, UINT64_MAX, mode, words[3],
                                 words[0], words[1], words[2]),
                          GOC_SUCCESS);
                for (unsigned lane = 0; lane < width; ++lane)
                  hash = goc_test::capture_hash_word(hash, words[3][lane]);
              }
            }
            EXPECT_EQ(hash, goc_test::fma32_exact_hashes[accumulate][boundary][policy])
                << accumulate << '/' << boundary << '/' << policy << '/' << wide << '/' << cpu;
          }
}

TEST(FmaExact, AllModifiersArchitecturesMasksAndAliases) {
  for (bool rdna3 : {false, true})
    for (bool wide : {false, true})
      for (bool accumulate : {false, true})
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
          for (uint32_t mode = 0; mode < 512; ++mode) {
            if (accumulate && (mode & (GOC_ALU_NEG_C | GOC_ALU_ABS_C)))
              continue;
            uint64_t flags =
                cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT | fp_flags(mode % 8);
            uint64_t exec_mask = mode & 1 ? 0xa7109efae12ce36dULL : UINT64_MAX;
            unsigned dest = mode % 4;
            uint32_t words[4][64], before[4][64];
            for (unsigned reg = 0; reg < 4; ++reg)
              for (unsigned lane = 0; lane < 64; ++lane)
                words[reg][lane] = edge[(lane + reg * 7) % 16];
            std::memcpy(before, words, sizeof(words));
            ASSERT_EQ(invoke(rdna3, wide, accumulate, flags, exec_mask, mode, words[dest], words[0],
                             words[1], words[2]),
                      GOC_SUCCESS);
            for (unsigned reg = 0; reg < 4; ++reg)
              for (unsigned lane = 0; lane < 64; ++lane) {
                uint32_t expected = before[reg][lane];
                if (reg == dest && lane < (wide ? 64U : 32U) && ((exec_mask >> lane) & 1))
                  expected = reference(before[0][lane], before[1][lane],
                                       before[accumulate ? dest : 2][lane], mode, flags, rdna3);
                EXPECT_EQ(words[reg][lane], expected);
              }
          }
}

TEST(FmaExact, DppMasksAliasingAndHostEnvironment) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (bool rdna3 : {false, true})
    for (bool wide : {false, true})
      for (bool accumulate : {false, true})
        for (unsigned policy = 0; policy < 8; ++policy)
          for (uint64_t descriptor : goc_test::dpp_modes) {
            ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
#if defined(__x86_64__) || defined(_M_X64)
            _mm_setcsr(_mm_getcsr() & ~0x8040U);
#endif
            uint64_t flags = (policy % (goc_init_cpu_flags() + 1)) | GOC_SEMANTICS_EXACT_EMPIRICAL |
                             GOC_SEMANTICS_STRICT | fp_flags(policy);
            uint64_t mode = descriptor | GOC_ALU_NEG_A | GOC_ALU_ABS_B | ((policy & 3) << 6) |
                            (policy & 4 ? GOC_ALU_CLAMP : 0);
            uint64_t exec_mask = wide ? 0xa7109efae12ce36dULL : 0xe12ce36dULL;
            unsigned dest = policy % 4;
            uint32_t before[4][64], expected[64];
            for (unsigned reg = 0; reg < 4; ++reg)
              for (unsigned lane = 0; lane < 64; ++lane)
                before[reg][lane] = edge[(lane + 7 * reg) % 16];
            for (unsigned lane = 0; lane < 64; ++lane) {
              expected[lane] = before[dest][lane];
              int source;
              if (lane < (wide ? 64U : 32U) && goc_test::dpp_source(mode, exec_mask, lane, source))
                expected[lane] =
                    reference(source < 0 ? 0 : before[0][source], before[1][lane],
                              before[accumulate ? dest : 2][lane], uint32_t(mode), flags, rdna3);
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
                uint32_t words[4][64];
                std::memcpy(words, before, sizeof(words));
                ASSERT_EQ(invoke(rdna3, wide, accumulate, flags, exec_mask, mode, words[dest],
                                 words[0], words[1], words[2]),
                          GOC_SUCCESS);
                for (unsigned reg = 0; reg < 4; ++reg)
                  for (unsigned lane = 0; lane < 64; ++lane)
                    EXPECT_EQ(words[reg][lane], reg == dest ? expected[lane] : before[reg][lane]);
                EXPECT_EQ(std::fegetround(), rounding);
                EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
#if defined(__x86_64__) || defined(_M_X64)
                EXPECT_EQ(_mm_getcsr(), controls);
#endif
              }
          }
}

TEST(FmaExact, ReportingRemainsExplicitlyUnsupported) {
  for (bool rdna3 : {false, true})
    for (bool wide : {false, true})
      for (bool accumulate : {false, true})
        for (uint64_t exec_mask : {uint64_t(0), UINT64_MAX})
          for (uint64_t strict : {uint64_t(0), GOC_SEMANTICS_STRICT}) {
            uint32_t state = 0x12345678;
            EXPECT_EQ(invoke(rdna3, wide, accumulate, GOC_SEMANTICS_EXACT_EMPIRICAL | strict,
                             exec_mask, 0, nullptr, nullptr, nullptr, nullptr, &state),
                      GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
            EXPECT_EQ(state, 0x12345678U);
          }
}

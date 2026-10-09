// SPDX-License-Identifier: MIT

#include "dpp_reference.h"
#include "exec_masks.h"
#include "fp_environment.h"
#include "goc/goc.h"

#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_v_fma_f16);

int fmac(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
         const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *,
         uint32_t *exceptions) {
  return goc_v_fmac_f16(flags, exec_mask, mode, d, a, b, exceptions);
}

const Fn functions[] = {goc_v_fma_f16, fmac, goc_v_div_fixup_f16};
const uint64_t exact = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;
const uint32_t values[] = {0x7c013c00, 0x00017bff, 0x4000fc00, 0x7e003c00,
                           0x80010000, 0x03ff0400, 0x3bffbc00, 0x7bff0001};

} // namespace

TEST(ArithmeticExceptions, DppMasksHalfSelectorsAndAliases) {
  for (Fn fn : functions)
    for (uint64_t descriptor : goc_test::dpp_modes)
      for (uint32_t exec_mask : exec_masks())
        for (uint32_t mode : {0U, GOC_ALU_HIGH_A, GOC_ALU_HIGH_B, GOC_ALU_HIGH_D, GOC_ALU_NEG_A,
                              GOC_ALU_ABS_B, GOC_ALU_OMOD_2, GOC_ALU_CLAMP})
          for (unsigned destination = 0; destination < 4; ++destination) {
            uint32_t initial[4][32];
            for (unsigned reg = 0; reg < 4; ++reg)
              for (unsigned lane = 0; lane < 32; ++lane)
                initial[reg][lane] = values[(lane + reg * 3) % 8];
            uint32_t expected_flags = 0x80000000, expected[32];
            std::memcpy(expected, initial[destination], sizeof(expected));
            for (unsigned lane = 0; lane < 32; ++lane) {
              int source;
              if (!goc_test::dpp_source(descriptor, exec_mask, lane, source))
                continue;
              uint32_t words[4][32] = {};
              words[0][0] = source < 0 ? 0 : initial[0][source];
              words[1][0] = initial[1][lane];
              words[2][0] = initial[2][lane];
              words[3][0] = initial[destination][lane];
              const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
              uint32_t *d[] = {words[3]};
              ASSERT_EQ(fn(exact, 1, mode, d, a, b, c, &expected_flags), GOC_SUCCESS);
              expected[lane] = words[3][0];
            }
            for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
              uint32_t words[4][32];
              std::memcpy(words, initial, sizeof(words));
              const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
              uint32_t *d[] = {words[destination]};
              uint32_t exceptions = 0x80000000;
              ASSERT_EQ(fn(cpu | exact, exec_mask, descriptor | mode, d, a, b, c, &exceptions),
                        GOC_SUCCESS);
              EXPECT_EQ(exceptions, expected_flags)
                  << descriptor << "/" << exec_mask << "/" << mode;
              for (unsigned lane = 0; lane < 32; ++lane)
                EXPECT_EQ(words[destination][lane], expected[lane]);
            }
          }
}

TEST(ArithmeticExceptions, HostEnvironmentAndOptOut) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (Fn fn : functions)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
        SCOPED_TRACE(fn == functions[0] ? "fma" : fn == functions[1] ? "fmac" : "fixup");
        SCOPED_TRACE(cpu);
        ASSERT_EQ(std::fesetround(rounding), 0);
#if defined(__x86_64__) || defined(_M_X64)
        // FTZ/DAZ must not change the integer-based guest classification.
        for (unsigned denorm_control : {0U, 0x8040U}) {
          _mm_setcsr((_mm_getcsr() & ~0x8040U) | denorm_control);
#endif
          uint32_t words[4][32];
          for (unsigned reg = 0; reg < 4; ++reg)
            for (unsigned lane = 0; lane < 32; ++lane)
              words[reg][lane] = values[(lane + reg * 3) % 8];
          const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
          uint32_t *d[] = {words[3]};
          uint32_t exceptions = 0x80000000;
          ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
          ASSERT_EQ(std::feraiseexcept(FE_DIVBYZERO), 0);
          ASSERT_EQ(fn(cpu | exact, UINT32_MAX, 0, d, a, b, c, &exceptions), GOC_SUCCESS);
          EXPECT_NE(exceptions, 0x80000000);
          EXPECT_EQ(std::fegetround(), rounding);
          // Compiler-generated arithmetic may raise additional sticky flags.
          EXPECT_EQ(std::fetestexcept(FE_DIVBYZERO), FE_DIVBYZERO);
#if defined(__x86_64__) || defined(_M_X64)
          EXPECT_EQ(_mm_getcsr() & 0x8040U, denorm_control);
        }
#endif
        uint32_t exceptions = 0x80000000;
        EXPECT_EQ(fn(cpu | exact, 0, 0, nullptr, nullptr, nullptr, nullptr, &exceptions),
                  GOC_SUCCESS);
        EXPECT_EQ(exceptions, 0x80000000);
        EXPECT_EQ(fn(cpu | exact, UINT32_MAX, 1ULL << 31, nullptr, nullptr, nullptr, nullptr,
                     &exceptions),
                  GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(exceptions, 0x80000000);
      }
}

TEST(ArithmeticExceptions, LooseModeSkipsReporting) {
  for (Fn fn : {goc_v_fma_f16, fmac, goc_v_div_fixup_f16, goc_v_pk_fma_f16})
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t words[5][32];
      for (unsigned reg = 0; reg < 5; ++reg)
        for (unsigned lane = 0; lane < 32; ++lane)
          words[reg][lane] = values[(lane + reg * 3) % 8];
      std::memcpy(words[4], words[3], sizeof(words[3]));
      const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
      uint32_t *d[] = {words[3]}, *expected[] = {words[4]};
      uint32_t exceptions = 0x80000000;
      ASSERT_EQ(fn(cpu, UINT32_MAX, 0, d, a, b, c, &exceptions), GOC_SUCCESS);
      ASSERT_EQ(fn(cpu, UINT32_MAX, 0, expected, a, b, c, nullptr), GOC_SUCCESS);
      EXPECT_EQ(exceptions, 0x80000000);
      for (unsigned lane = 0; lane < 32; ++lane)
        EXPECT_EQ(words[3][lane], words[4][lane]);
    }
}

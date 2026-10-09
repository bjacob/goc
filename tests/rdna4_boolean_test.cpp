// SPDX-License-Identifier: MIT

#include "fp_environment.h"
#include "goc/goc.h"
#include "rdna4_boolean_reference.h"
#include "rdna4_exec_masks.h"

#include <algorithm>
#include <cfenv>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace {

::testing::AssertionResult check(int op, uint64_t flags, uint32_t mode, uint32_t words[3][32]) {
  uint32_t expected[32];
  for (int lane = 0; lane < 32; ++lane)
    expected[lane] =
        goc_test::boolean_reference(op, words[0][lane], words[1][lane], words[2][lane], mode);
  const uint32_t *a[] = {words[0]}, *b[] = {words[1]};
  uint32_t *d[] = {words[2]};
  int status = goc_test::boolean_functions[op](flags, UINT32_MAX, mode, d, a, b);
  if (status != GOC_SUCCESS)
    return ::testing::AssertionFailure() << "status " << status;
  for (int lane = 0; lane < 32; ++lane)
    if (words[2][lane] != expected[lane])
      return ::testing::AssertionFailure() << op << "/" << flags << "/" << mode << "/" << lane;
  return ::testing::AssertionSuccess();
}

} // namespace

TEST(Boolean, TruthTablesAtEveryBitAndSelector) {
  for (int op = 0; op < 9; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned selection = 0; selection < ((op < 4 || op == 8) ? 1u : 8u); ++selection)
        for (unsigned truth = 0; truth < 4; ++truth)
          for (bool surrounding : {false, true}) {
            uint32_t words[3][32];
            for (int lane = 0; lane < 32; ++lane) {
              uint32_t bit = uint32_t(1) << lane;
              words[0][lane] = (surrounding ? ~bit : 0) | (truth & 1 ? bit : 0);
              words[1][lane] = (surrounding ? ~bit : 0) | (truth & 2 ? bit : 0);
              words[2][lane] = 0x5a39c861;
            }
            ASSERT_TRUE(check(op, cpu, goc_test::boolean_mode(op, selection), words));
          }
}

TEST(Boolean, EveryHalfEncodingAndRandomFullWords) {
  for (int op = 0; op < 9; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned selection = 0; selection < ((op < 4 || op == 8) ? 1u : 8u); ++selection) {
        std::mt19937 random(9384);
        for (unsigned start = 0; start < 65536; start += 32) {
          uint32_t words[3][32];
          for (unsigned lane = 0; lane < 32; ++lane) {
            unsigned value = start + lane;
            words[0][lane] = (op < 4 || op == 8) ? random() : value | ((65535 - value) << 16);
            words[1][lane] = random();
            words[2][lane] = random();
          }
          ASSERT_TRUE(check(op, cpu, goc_test::boolean_mode(op, selection), words));
        }
      }
}

TEST(Boolean, MasksAliasesAndUnalignedStorage) {
  for (int op = 0; op < 9; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned selection = 0; selection < ((op < 4 || op == 8) ? 1u : 8u); ++selection)
        for (bool same_sources : {false, true}) {
          std::mt19937 random(678);
          uint32_t original[3][35];
          for (auto &reg : original)
            for (auto &word : reg)
              word = random();
          uint32_t mode = goc_test::boolean_mode(op, selection);
          int breg = same_sources ? 0 : 1;
          for (uint32_t mask : rdna4_exec_masks())
            for (int target = 0; target < 3; ++target) {
              uint32_t words[3][35], expected[3][35];
              for (int reg = 0; reg < 3; ++reg) {
                std::copy_n(original[reg], 35, words[reg]);
                std::copy_n(original[reg], 35, expected[reg]);
              }
              for (int lane = 0; lane < 32; ++lane)
                if ((mask >> lane) & 1)
                  expected[target][lane + 1] = goc_test::boolean_reference(
                      op, original[0][lane + 1], original[breg][lane + 1],
                      original[target][lane + 1], mode);
              const uint32_t *a[] = {words[0] + 1}, *b[] = {words[breg] + 1};
              uint32_t *d[] = {words[target] + 1};
              ASSERT_EQ(goc_test::boolean_functions[op](cpu, mask, mode, d, a, b), GOC_SUCCESS);
              for (int reg = 0; reg < 3; ++reg)
                ASSERT_TRUE(std::equal(words[reg], words[reg] + 35, expected[reg]));
            }
        }
}

TEST(Boolean, ValidationBeforeEmptyMaskAndUnchangedOutputs) {
  for (int op = 0; op < 9; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t words[3][32];
      for (auto &reg : words)
        std::fill_n(reg, 32, 0x859e43c1);
      const uint32_t *a[] = {words[0]}, *b[] = {words[1]};
      uint32_t *d[] = {words[2]};
      uint32_t known = goc_test::boolean_mode(op, 7);
      for (uint32_t mask : {0U, UINT32_MAX}) {
        for (int bit = 0; bit < 32; ++bit)
          if (!(known & (uint32_t(1) << bit))) {
            EXPECT_EQ(goc_test::boolean_functions[op](cpu, mask, uint32_t(1) << bit, d, a, b),
                      GOC_ERROR_INVALID_FLAGS);
          }
        EXPECT_EQ(goc_test::boolean_functions[op](cpu | (1ULL << 63), mask, known, d, a, b),
                  GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(goc_test::boolean_functions[op](cpu | GOC_SEMANTICS_EXACT_EMPIRICAL |
                                                      GOC_SEMANTICS_STRICT,
                                                  mask, known, d, a, b),
                  GOC_ERROR_UNSUPPORTED_SEMANTICS);
      }
      for (const auto &reg : words)
        for (uint32_t value : reg)
          EXPECT_EQ(value, 0x859e43c1);
      EXPECT_TRUE(check(op, cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_FP16_OVFL, known, words));
    }
}

TEST(Boolean, HostFpStateIsPreserved) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO})
    for (int flush = 0; flush < 2; ++flush) {
      std::fesetround(rounding);
      std::feclearexcept(FE_ALL_EXCEPT);
      std::feraiseexcept(FE_DIVBYZERO);
#if defined(__x86_64__) || defined(_M_X64)
      _mm_setcsr((_mm_getcsr() & ~0x8040u) | (flush ? 0x8040u : 0));
      unsigned before = _mm_getcsr();
#endif
      for (int op = 0; op < 9; ++op)
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
          for (unsigned selection = 0; selection < ((op < 4 || op == 8) ? 1u : 8u); ++selection) {
            uint32_t words[3][32];
            for (auto &reg : words)
              std::fill_n(reg, 32, 0x7f800001);
            EXPECT_TRUE(check(op, cpu, goc_test::boolean_mode(op, selection), words));
          }
      EXPECT_EQ(std::fegetround(), rounding);
      EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
#if defined(__x86_64__) || defined(_M_X64)
      EXPECT_EQ(_mm_getcsr(), before);
#endif
    }
}

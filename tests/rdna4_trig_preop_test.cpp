// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "rdna4_exec_masks.h"
#include "rdna4_trig_preop_hardware.h"

#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>

TEST(TrigPreop, HardwareEveryExponentSelectorAndModifier) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned mode = 0; mode < 32; ++mode) {
      uint64_t digest = goc_test::capture_hash_seed;
      for (unsigned e = 0; e < 2048; ++e) {
        uint32_t words[5][32];
        for (unsigned lane = 0; lane < 32; ++lane) {
          uint32_t i = e * 32 + lane;
          uint64_t fraction =
              (uint64_t(i * 0x9e3779b9u) * 0xd1b54a32d192ed03ULL) & 0xfffffffffffffULL;
          words[0][lane] = uint32_t(fraction);
          words[1][lane] = (e << 20) | ((lane & 1) << 31) | uint32_t(fraction >> 32);
          words[2][lane] = lane | 0xdeadbee0;
        }
        const uint32_t *a[] = {words[0], words[1]}, *b[] = {words[2]};
        uint32_t *d[] = {words[3], words[4]};
        ASSERT_EQ(goc_rdna4_v_trig_preop_f64(
                      cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, UINT32_MAX,
                      goc_test::trig_preop_mode(mode), d, a, b, nullptr),
                  GOC_SUCCESS);
        for (unsigned lane = 0; lane < 32; ++lane)
          for (unsigned reg = 0; reg < 2; ++reg)
            digest = goc_test::capture_hash_bytes(digest, d[reg][lane], 4);
      }
      EXPECT_EQ(digest, goc_test::trig_preop_digests[mode]) << cpu << "/" << mode;
    }
}

TEST(TrigPreop, MasksAliasesAndUnalignedStorage) {
  uint32_t initial[5][35];
  for (unsigned reg = 0; reg < 5; ++reg)
    for (unsigned lane = 0; lane < 35; ++lane)
      initial[reg][lane] = 0xaabbee11u + lane;
  for (unsigned lane = 0; lane < 32; ++lane) {
    initial[1][lane + 1] = goc_test::trig_preop_exponent(lane) << 20;
    initial[2][lane + 1] = goc_test::trig_preop_selector(lane);
  }
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned mode = 0; mode < 32; ++mode)
      for (unsigned target = 0; target < 25; ++target)
        for (uint32_t exec_mask : rdna4_exec_masks()) {
          uint32_t words[5][35];
          std::memcpy(words, initial, sizeof(words));
          const uint32_t *a[] = {words[0] + 1, words[1] + 1}, *b[] = {words[2] + 1};
          uint32_t *d[] = {words[target / 5] + 1, words[target % 5] + 1};
          ASSERT_EQ(goc_rdna4_v_trig_preop_f64(cpu, exec_mask, goc_test::trig_preop_mode(mode), d,
                                               a, b, nullptr),
                    GOC_SUCCESS);
          for (unsigned reg = 0; reg < 5; ++reg)
            for (unsigned lane = 0; lane < 35; ++lane) {
              uint32_t want = initial[reg][lane];
              if (lane > 0 && lane <= 32 && ((exec_mask >> (lane - 1)) & 1)) {
                uint64_t value = goc_test::trig_preop_samples[mode][lane - 1];
                if (reg == target / 5)
                  want = uint32_t(value);
                if (reg == target % 5)
                  want = uint32_t(value >> 32);
              }
              ASSERT_EQ(words[reg][lane], want)
                  << cpu << "/" << mode << "/" << target << "/" << lane;
            }
        }
}

TEST(TrigPreop, PreservesFpStateAndValidatesFlags) {
  {
    goc_test::ScopedFpEnvironment saved;
    ASSERT_TRUE(saved.saved());
    uint32_t a0[32]{}, a1[32], b0[32]{}, d0[32], d1[32];
    const uint32_t *a[] = {a0, a1}, *b[] = {b0};
    uint32_t *d[] = {d0, d1};
    for (unsigned lane = 0; lane < 32; ++lane)
      a1[lane] = 0x7ffabcde;
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        std::fesetround(rounding);
        std::feclearexcept(FE_ALL_EXCEPT);
        std::feraiseexcept(FE_INVALID | FE_INEXACT);
        int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
        EXPECT_EQ(goc_rdna4_v_trig_preop_f64(cpu, UINT32_MAX, goc_test::trig_preop_mode(31), d, a,
                                             b, nullptr),
                  GOC_SUCCESS);
        EXPECT_EQ(std::fegetround(), rounding);
        EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
      }
  }
  EXPECT_EQ(goc_rdna4_v_trig_preop_f64(0, 0U, 0, nullptr, nullptr, nullptr, nullptr), GOC_SUCCESS);
  uint32_t known = goc_test::trig_preop_mode(31);
  for (unsigned bit = 0; bit < 32; ++bit)
    if (!(known & (1u << bit))) {
      EXPECT_EQ(goc_rdna4_v_trig_preop_f64(0, 0, 1u << bit, nullptr, nullptr, nullptr, nullptr),
                GOC_ERROR_INVALID_FLAGS);
    }
  EXPECT_EQ(goc_rdna4_v_trig_preop_f64(1ULL << 63, 0, 0, nullptr, nullptr, nullptr, nullptr),
            GOC_ERROR_INVALID_FLAGS);
}

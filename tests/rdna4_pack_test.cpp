// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "rdna4_dpp_reference.h"
#include "rdna4_exec_masks.h"
#include "rdna4_pack_reference.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

TEST(Pack, ExhaustiveHardwareEncodingsAndModifiers) {
  for (unsigned variant = 0; variant < 66; ++variant)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint64_t digest = goc_test::capture_hash_seed;
      for (unsigned start = 0; start < 65536; start += 32) {
        uint32_t words[3][32];
        for (unsigned lane = 0; lane < 32; ++lane) {
          unsigned i = start + lane;
          words[0][lane] = i | ((65535 - i) << 16);
          words[1][lane] = (i * 0x7395a831u) ^ 0xa7925163u;
          words[2][lane] = 0xcafebeef;
        }
        const uint32_t *a[] = {words[0]}, *b[] = {words[1]};
        uint32_t *d[] = {words[2]};
        ASSERT_EQ(
            goc_test::pack_call(variant, cpu, UINT32_MAX, goc_test::pack_mode(variant), d, a, b),
            GOC_SUCCESS);
        for (uint32_t word : words[2])
          digest = goc_test::capture_hash_bytes(digest, word, 4);
      }
      EXPECT_EQ(digest, goc_test::pack_capture_digests[variant]) << variant << "/" << cpu;
    }
}

TEST(Pack, MasksAliasesAndUnalignedStorage) {
  std::mt19937 random(75614);
  uint32_t initial[3][35];
  for (auto &reg : initial)
    for (auto &word : reg)
      word = random();
  for (unsigned variant = 0; variant < 66; ++variant)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned breg : {0u, 1u})
        for (unsigned target = 0; target < 3; ++target)
          for (uint32_t exec_mask : rdna4_exec_masks()) {
            uint32_t words[3][35], expected[3][35];
            std::memcpy(words, initial, sizeof(words));
            std::memcpy(expected, initial, sizeof(expected));
            for (unsigned lane = 0; lane < 32; ++lane)
              if ((exec_mask >> lane) & 1)
                expected[target][lane + 1] =
                    goc_test::pack_reference(variant, initial[0][lane + 1], initial[breg][lane + 1],
                                             initial[target][lane + 1]);
            const uint32_t *a[] = {words[0] + 1}, *b[] = {words[breg] + 1};
            uint32_t *d[] = {words[target] + 1};
            ASSERT_EQ(
                goc_test::pack_call(variant, cpu, exec_mask, goc_test::pack_mode(variant), d, a, b),
                GOC_SUCCESS);
            ASSERT_EQ(std::memcmp(words, expected, sizeof(words)), 0) << variant << "/" << cpu;
          }
}

TEST(Pack, ValidationAndEmptyMask) {
  for (unsigned variant : {0u, 2u}) {
    uint32_t allowed = goc_test::pack_mode(variant == 0 ? 1 : 65);
    EXPECT_EQ(goc_test::pack_call(variant, 0, 0U, 0, nullptr, nullptr, nullptr), GOC_SUCCESS);
    for (unsigned bit = 0; bit < 32; ++bit) {
      if (!(allowed & (1u << bit))) {
        EXPECT_EQ(goc_test::pack_call(variant, 0, 0, 1u << bit, nullptr, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
      }
    }
    EXPECT_EQ(goc_test::pack_call(variant, GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0,
                                  0, nullptr, nullptr, nullptr),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
  }
}

TEST(Pack, HostFpStatePreservedForSignalingNaNs) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    std::fesetround(rounding);
    std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_DIVBYZERO);
    for (unsigned variant = 0; variant < 66; ++variant)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
        uint32_t words[3][32];
        for (auto &reg : words)
          std::fill_n(reg, 32, 0xfc017c01);
        const uint32_t *a[] = {words[0]}, *b[] = {words[1]};
        uint32_t *d[] = {words[2]};
        EXPECT_EQ(
            goc_test::pack_call(variant, cpu, UINT32_MAX, goc_test::pack_mode(variant), d, a, b),
            GOC_SUCCESS);
        EXPECT_EQ(std::fegetround(), rounding);
        EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
      }
  }
}

TEST(Pack, DppModifiersMasksAliasesAndGuards) {
  std::mt19937 random(827619);
  uint32_t initial[3][34];
  for (auto &reg : initial)
    for (auto &word : reg)
      word = random();
  for (unsigned variant = 0; variant < 66; ++variant)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (auto descriptor : goc_test::dpp_modes)
        for (auto exec_mask : rdna4_exec_masks())
          for (unsigned target = 0; target < 3; ++target) {
            uint32_t words[3][34], expected[3][34];
            std::memcpy(words, initial, sizeof(words));
            std::memcpy(expected, initial, sizeof(expected));
            for (unsigned lane = 0; lane < 32; ++lane) {
              int source = 0;
              if (goc_test::dpp_source(descriptor, exec_mask, lane, source))
                expected[target][lane + 1] =
                    goc_test::pack_reference(variant, source < 0 ? 0 : initial[0][source + 1],
                                             initial[1][lane + 1], initial[target][lane + 1]);
            }
            const uint32_t *a[] = {words[0] + 1}, *b[] = {words[1] + 1};
            uint32_t *d[] = {words[target] + 1};
            ASSERT_EQ(goc_test::pack_call(variant, cpu, exec_mask,
                                          descriptor | goc_test::pack_mode(variant), d, a, b),
                      GOC_SUCCESS);
            ASSERT_EQ(std::memcmp(words, expected, sizeof(words)), 0)
                << variant << "/" << cpu << "/" << descriptor << "/" << exec_mask;
          }
}

TEST(Pack, DppValidation) {
  for (unsigned variant : {0u, 2u})
    for (auto descriptor : goc_test::dpp_modes) {
      EXPECT_EQ(goc_test::pack_call(variant, 0, 0, descriptor, nullptr, nullptr, nullptr),
                GOC_SUCCESS);
      EXPECT_EQ(
          goc_test::pack_call(variant, 0, 0, descriptor | (1ULL << 36), nullptr, nullptr, nullptr),
          GOC_ERROR_INVALID_FLAGS);
    }
}

// RX 9070 captures: eight EXEC masks, all 66 packing modes, seven DPP
// descriptors and 32 lanes. The digest includes both destination halves.
TEST(Pack, DppHardwareCorpusAndHostFpState) {
  const uint32_t values[] = {0x000100ff, 0x0100ffff, 0x80007fff, 0xfc017c01,
                             0x00008000, 0x7c00fc00, 0x3c00bc00, 0x7fff0001};
  const uint32_t masks[] = {0xffffffff, 0,          0xaaaaaaaa, 0x55555555,
                            1,          0x80000000, 0xffff,     0xffff0000};
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    std::fesetround(rounding);
    std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_DIVBYZERO);
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint64_t hash = goc_test::capture_hash_seed;
      for (auto exec_mask : masks)
        for (unsigned variant = 0; variant < 66; ++variant)
          for (auto descriptor : goc_test::dpp_modes) {
            uint32_t av[32], bv[32], output[32];
            for (unsigned lane = 0; lane < 32; ++lane) {
              av[lane] = values[lane % 8];
              bv[lane] = values[(lane + 3) % 8];
              output[lane] = 0xdead0000u + lane;
            }
            const uint32_t *a[] = {av}, *b[] = {bv};
            uint32_t *d[] = {output};
            EXPECT_EQ(goc_test::pack_call(variant, cpu, exec_mask,
                                          descriptor | goc_test::pack_mode(variant), d, a, b),
                      GOC_SUCCESS);
            for (auto word : output)
              hash = goc_test::capture_hash_word(hash, word);
          }
      EXPECT_EQ(hash, 0x001666e8f93d98c9ULL) << cpu;
      EXPECT_EQ(std::fegetround(), rounding);
      EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
    }
  }
}

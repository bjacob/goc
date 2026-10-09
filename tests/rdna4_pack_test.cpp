// SPDX-License-Identifier: MIT

#include "goc/goc.h"
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
      uint64_t digest = UINT64_C(14695981039346656037);
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
          for (unsigned byte = 0; byte < 4; ++byte) {
            digest ^= (word >> (8 * byte)) & 255;
            digest *= UINT64_C(1099511628211);
          }
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
          for (uint64_t mask : rdna4_exec_masks()) {
            uint32_t words[3][35], expected[3][35];
            std::memcpy(words, initial, sizeof(words));
            std::memcpy(expected, initial, sizeof(expected));
            for (unsigned lane = 0; lane < 32; ++lane)
              if ((mask >> lane) & 1)
                expected[target][lane + 1] =
                    goc_test::pack_reference(variant, initial[0][lane + 1], initial[breg][lane + 1],
                                             initial[target][lane + 1]);
            const uint32_t *a[] = {words[0] + 1}, *b[] = {words[breg] + 1};
            uint32_t *d[] = {words[target] + 1};
            ASSERT_EQ(
                goc_test::pack_call(variant, cpu, mask, goc_test::pack_mode(variant), d, a, b),
                GOC_SUCCESS);
            ASSERT_EQ(std::memcmp(words, expected, sizeof(words)), 0) << variant << "/" << cpu;
          }
}

TEST(Pack, ValidationAndEmptyMask) {
  for (unsigned variant : {0u, 2u}) {
    uint32_t allowed = goc_test::pack_mode(variant == 0 ? 1 : 65);
    EXPECT_EQ(
        goc_test::pack_call(variant, 0, UINT64_C(0xffffffff00000000), 0, nullptr, nullptr, nullptr),
        GOC_SUCCESS);
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
  std::fenv_t saved;
  ASSERT_EQ(std::fegetenv(&saved), 0);
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
  std::fesetenv(&saved);
}

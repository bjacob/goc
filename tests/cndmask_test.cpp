// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "cndmask_hardware.h"
#include "cndmask_reference.h"
#include "dpp_reference.h"
#include "exec_masks.h"
#include "fp_environment.h"
#include "goc/goc.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>
#include <vector>

namespace {

using Fn = decltype(&goc_v_cndmask_b32);
const Fn functions[] = {goc_v_cndmask_b32, goc_v_cndmask_b16};

} // namespace

TEST(Cndmask, HardwareAllEncodingsAndModifiers) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned variant = 0; variant < 288; ++variant) {
      bool half = variant >= 32;
      unsigned m = (half ? variant - 32 : variant) / 2;
      uint32_t condition = variant & 1 ? UINT32_MAX : 0;
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
        ASSERT_EQ(functions[half](cpu, UINT32_MAX, goc_test::cndmask_mode(m), d, a, b, condition),
                  GOC_SUCCESS);
        for (uint32_t value : words[2])
          digest = goc_test::capture_hash_bytes(digest, value, 4);
      }
      EXPECT_EQ(digest, goc_test::cndmask_digests[variant]) << cpu << "/" << variant;
    }
}

TEST(Cndmask, EveryModifierMaskAliasAndUnalignedStorage) {
  for (unsigned half = 0; half < 2; ++half)
    for (unsigned m = 0; m < (half ? 128u : 16u); ++m)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (unsigned alias = 0; alias < 2; ++alias)
          for (unsigned target = 0; target < 3; ++target)
            for (uint32_t exec_mask : exec_masks()) {
              uint32_t words[3][35], expected[3][35];
              for (unsigned reg = 0; reg < 3; ++reg)
                for (unsigned lane = 0; lane < 35; ++lane)
                  words[reg][lane] = expected[reg][lane] =
                      (lane * 0x7395a831u) ^ (reg * 0xa7925163u);
              uint32_t condition = (m * 0x9e3779b9u) ^ 0x96969696u;
              unsigned source_b = alias ? 0 : 1;
              for (unsigned lane = 0; lane < 32; ++lane)
                if ((exec_mask >> lane) & 1)
                  expected[target][lane + 1] = goc_test::cndmask_reference(
                      half, words[0][lane + 1], words[source_b][lane + 1], words[target][lane + 1],
                      m, (condition >> lane) & 1);
              const uint32_t *a[] = {words[0] + 1}, *b[] = {words[source_b] + 1};
              uint32_t *d[] = {words[target] + 1};
              ASSERT_EQ(
                  functions[half](cpu, exec_mask, goc_test::cndmask_mode(m), d, a, b, condition),
                  GOC_SUCCESS);
              for (unsigned reg = 0; reg < 3; ++reg)
                ASSERT_TRUE(std::equal(words[reg], words[reg] + 35, expected[reg]))
                    << half << "/" << m << "/" << cpu << "/" << target;
            }
}

TEST(Cndmask, EveryConditionBitIndependentOfExec) {
  for (unsigned half = 0; half < 2; ++half)
    for (unsigned m = 0; m < (half ? 128u : 16u); ++m)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (uint64_t condition : exec_masks()) {
          uint32_t words[3][32];
          for (unsigned lane = 0; lane < 32; ++lane) {
            words[0][lane] = 0x7c017f81u + lane;
            words[1][lane] = 0xff818123u - lane;
            words[2][lane] = 0x12345678;
          }
          const uint32_t *a[] = {words[0]}, *b[] = {words[1]};
          uint32_t *d[] = {words[2]};
          uint32_t exec_mask = (m & 1) ? 0xaaaaaaaa : 0x55555555;
          ASSERT_EQ(functions[half](cpu, exec_mask, goc_test::cndmask_mode(m), d, a, b,
                                    uint32_t(condition)),
                    GOC_SUCCESS);
          for (unsigned lane = 0; lane < 32; ++lane) {
            uint32_t want =
                (exec_mask >> lane) & 1
                    ? goc_test::cndmask_reference(half, words[0][lane], words[1][lane], 0x12345678,
                                                  m, (condition >> lane) & 1)
                    : 0x12345678;
            ASSERT_EQ(words[2][lane], want) << half << "/" << m << "/" << cpu << "/" << lane;
          }
        }
}

TEST(Cndmask, InvalidFlagsAndHostFpState) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (unsigned half = 0; half < 2; ++half) {
    uint32_t known = goc_test::cndmask_mode(half ? 127 : 15);
    EXPECT_EQ(functions[half](0, 0U, known, nullptr, nullptr, nullptr, 0), GOC_SUCCESS);
    for (unsigned bit = 0; bit < 32; ++bit) {
      if (!(known & (1u << bit))) {
        EXPECT_EQ(functions[half](0, 0, 1u << bit, nullptr, nullptr, nullptr, 0),
                  GOC_ERROR_INVALID_FLAGS);
      }
    }
    EXPECT_EQ(functions[half](GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0, nullptr,
                              nullptr, nullptr, 0),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        uint32_t a0[32], b0[32], d0[32];
        std::fill_n(a0, 32, 0x7f800001);
        std::fill_n(b0, 32, 0x7c018001);
        std::fill_n(d0, 32, 0xaabbccdd);
        const uint32_t *a[] = {a0}, *b[] = {b0};
        uint32_t *d[] = {d0};
        std::fesetround(rounding);
        std::feclearexcept(FE_ALL_EXCEPT);
        std::feraiseexcept(FE_INVALID | FE_DIVBYZERO);
        int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
        EXPECT_EQ(functions[half](cpu, UINT32_MAX, known, d, a, b, 0xaaaaaaaa), GOC_SUCCESS);
        EXPECT_EQ(std::fegetround(), rounding);
        EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
      }
  }
}

TEST(Cndmask, DppModifiersMasksAliasesAndGuards) {
  for (unsigned half = 0; half < 2; ++half)
    for (unsigned m = 0; m < (half ? 128u : 16u); ++m)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (auto descriptor : goc_test::dpp_modes) {
          auto masks = (m == 0 || m == (half ? 127u : 15u)) ? exec_masks()
                                                            : std::vector<uint32_t>{UINT32_MAX};
          for (auto exec_mask : masks)
            for (unsigned source_b : {0u, 1u})
              for (unsigned target = 0; target < 3; ++target) {
                uint32_t words[3][34], expected[3][34];
                for (unsigned reg = 0; reg < 3; ++reg)
                  for (unsigned word = 0; word < 34; ++word)
                    words[reg][word] = expected[reg][word] =
                        (word * 0x7395a831u) ^ (reg * 0xa7925163u);
                uint32_t condition = (m * 0x9e3779b9u) ^ 0x96969696u;
                for (unsigned lane = 0; lane < 32; ++lane) {
                  int source = 0;
                  if (goc_test::dpp_source(descriptor, exec_mask, lane, source))
                    expected[target][lane + 1] = goc_test::cndmask_reference(
                        half, source < 0 ? 0 : words[0][source + 1], words[source_b][lane + 1],
                        words[target][lane + 1], m, (condition >> lane) & 1);
                }
                const uint32_t *a[] = {words[0] + 1}, *b[] = {words[source_b] + 1};
                uint32_t *d[] = {words[target] + 1};
                ASSERT_EQ(functions[half](cpu, exec_mask, descriptor | goc_test::cndmask_mode(m), d,
                                          a, b, condition),
                          GOC_SUCCESS);
                ASSERT_EQ(std::memcmp(words, expected, sizeof(words)), 0)
                    << half << "/" << m << "/" << cpu << "/" << descriptor << "/" << exec_mask;
              }
        }
}

TEST(Cndmask, DppEveryConditionBitIndependentOfExec) {
  for (unsigned half = 0; half < 2; ++half)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (auto descriptor : goc_test::dpp_modes)
        for (auto condition : exec_masks())
          for (uint32_t exec_mask : {0xaaaaaaaau, 0x55555555u}) {
            unsigned m = half ? 127 : 15;
            uint32_t av[32], bv[32], output[32];
            for (unsigned lane = 0; lane < 32; ++lane) {
              av[lane] = 0xfc017f81u + lane;
              bv[lane] = 0x7c01ff81u - lane;
              output[lane] = 0x12345678;
            }
            const uint32_t *a[] = {av}, *b[] = {bv};
            uint32_t *d[] = {output};
            ASSERT_EQ(functions[half](cpu, exec_mask, descriptor | goc_test::cndmask_mode(m), d, a,
                                      b, uint32_t(condition)),
                      GOC_SUCCESS);
            for (unsigned lane = 0; lane < 32; ++lane) {
              int source = 0;
              uint32_t want = 0x12345678;
              if (goc_test::dpp_source(descriptor, exec_mask, lane, source))
                want = goc_test::cndmask_reference(half, source < 0 ? 0 : av[source], bv[lane],
                                                   want, m, (condition >> lane) & 1);
              ASSERT_EQ(output[lane], want)
                  << half << "/" << cpu << "/" << descriptor << "/" << lane;
            }
          }
}

TEST(Cndmask, DppValidation) {
  for (auto fn : functions)
    for (auto descriptor : goc_test::dpp_modes) {
      EXPECT_EQ(fn(0, 0, descriptor, nullptr, nullptr, nullptr, UINT32_MAX), GOC_SUCCESS);
      for (auto invalid : {1ULL << 36, 1ULL << 6})
        EXPECT_EQ(fn(0, 0, descriptor | invalid, nullptr, nullptr, nullptr, UINT32_MAX),
                  GOC_ERROR_INVALID_FLAGS);
    }
}

// RX 9070 capture: three condition masks, eight EXEC masks, every source
// modifier and half selector, seven DPP descriptors, 32 lanes per instruction.
TEST(Cndmask, DppHardwareCorpusAndHostFpState) {
  const uint32_t values[] = {0x00000001, 0x807fffff, 0x00800000, 0x80000000,
                             0x7f800001, 0xff800000, 0x3f800000, 0xff7fffff};
  const uint32_t masks[] = {0xffffffff, 0,          0xaaaaaaaa, 0x55555555,
                            1,          0x80000000, 0xffff,     0xffff0000};
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    std::fesetround(rounding);
    std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_INVALID | FE_DIVBYZERO);
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint64_t hash = goc_test::capture_hash_seed;
      for (uint32_t condition : {0u, UINT32_MAX, 0x96969696u})
        for (auto exec_mask : masks)
          for (unsigned half = 0; half < 2; ++half)
            for (unsigned m = 0; m < (half ? 128u : 16u); ++m)
              for (auto descriptor : goc_test::dpp_modes) {
                uint32_t av[32], bv[32], output[32];
                for (unsigned lane = 0; lane < 32; ++lane) {
                  av[lane] = values[lane % 8];
                  bv[lane] = values[(lane + 3) % 8];
                  output[lane] = 0xdead0000u + lane;
                }
                const uint32_t *a[] = {av}, *b[] = {bv};
                uint32_t *d[] = {output};
                EXPECT_EQ(functions[half](cpu, exec_mask, descriptor | goc_test::cndmask_mode(m), d,
                                          a, b, condition),
                          GOC_SUCCESS);
                for (auto word : output)
                  hash = goc_test::capture_hash_word(hash, word);
              }
      EXPECT_EQ(hash, 0x237a7524109b1925ULL) << cpu;
      EXPECT_EQ(std::fegetround(), rounding);
      EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_INVALID | FE_DIVBYZERO);
    }
  }
}

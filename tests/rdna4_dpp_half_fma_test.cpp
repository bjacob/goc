// SPDX-License-Identifier: MIT

#include "fp_environment.h"
#include "goc/goc.h"
#include "rdna4_dpp_half_fma_reference.h"
#include "rdna4_dpp_reference.h"
#include "rdna4_exec_masks.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>
#include <vector>

namespace {

std::vector<uint64_t> modes(unsigned op) {
  std::vector<uint64_t> result;
  const std::vector<unsigned> low_modes =
      op == 0
          ? std::vector<unsigned>{0,   1,   2,   4,    8,    16,   32,   64,   128,
                                  192, 256, 512, 1024, 2048, 4096, 7680, 8191, 7935}
          : std::vector<unsigned>{0, 1, 2, 8, 16, 64, 128, 192, 256, 512, 1024, 4096, 6107, 5851};
  for (uint64_t descriptor : goc_test::dpp_modes)
    for (unsigned low : low_modes)
      result.push_back(descriptor | low);
  return result;
}

uint32_t canonical(uint32_t word, uint32_t mode) {
  unsigned shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  if (((word >> shift) & 0x7fff) > 0x7c00)
    return (word & ~(UINT32_C(65535) << shift)) | (UINT32_C(0x7e00) << shift);
  return word;
}

void hardware_corpus(const uint32_t *values, uint64_t expected_hash) {
  // Gfx1201: 32 operation/modifier combinations x seven DPP descriptors
  // x eight EXEC masks x 32 lanes.
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (uint64_t sem : {UINT64_C(0), GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT}) {
      uint64_t hash = UINT64_C(14695981039346656037);
      for (uint32_t mask :
           {0xffffffffu, 0u, 0xaaaaaaaau, 0x55555555u, 1u, 0x80000000u, 0xffffu, 0xffff0000u})
        for (unsigned op = 0; op < 2; ++op)
          for (uint64_t mode : modes(op)) {
            uint32_t a[32], b[32], c[32], d[32];
            for (unsigned lane = 0; lane < 32; ++lane) {
              a[lane] = values[lane % 8];
              b[lane] = values[(lane + 3) % 8];
              c[lane] = values[(lane + 5) % 8];
              d[lane] = values[(lane + 1) % 8];
            }
            auto pa = a, pb = b, pc = c, pd = d;
            ASSERT_EQ(goc_test::half_fma_functions[op](cpu | sem, mask, mode, &pd, &pa, &pb, &pc),
                      GOC_SUCCESS);
            for (unsigned lane = 0; lane < 32; ++lane) {
              int source;
              uint32_t want = values[(lane + 1) % 8];
              if (goc_test::dpp_source(mode, mask, lane, source))
                want = goc_test::half_fma_result(op, source < 0 ? 0 : a[source], b[lane], c[lane],
                                                 uint32_t(mode), want);
              ASSERT_EQ(d[lane], want) << cpu << '/' << op << '/' << mode << '/' << lane;
              hash = (hash ^ d[lane]) * UINT64_C(1099511628211);
            }
          }
      EXPECT_EQ(hash, expected_hash);
    }
}

} // namespace

TEST(DppHalfFma, HardwareCorpus) {
  const uint32_t values[] = {0x3c00bc00u, 0xc0004000u, 0x38003400u, 0xb800b400u,
                             0x44004200u, 0xc400c200u, 0x3a003600u, 0xba00b600u};
  hardware_corpus(values, UINT64_C(0xdef3376de08ee575));
}

TEST(DppHalfFma, HardwareOmodBoundaries) {
  const uint32_t values[] = {0x00018001u, 0x03ff83ffu, 0x04008400u, 0x04018401u,
                             0x08008800u, 0x3c00bc00u, 0x00008000u, 0x7bfffbffu};
  hardware_corpus(values, UINT64_C(0xc762700924a242d2));
}

TEST(DppHalfFma, MasksAliasesAndRandomWords) {
  const unsigned layouts[][3] = {{0, 1, 2}, {0, 0, 2}, {0, 1, 0}, {0, 1, 1}, {0, 0, 0}};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (uint64_t sem : {UINT64_C(0), GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT})
      for (unsigned op = 0; op < 2; ++op)
        for (uint64_t mode : modes(op))
          for (uint32_t mask : rdna4_exec_masks())
            for (const auto &layout : layouts)
              for (unsigned dest = 0; dest < 4; ++dest) {
                SCOPED_TRACE(::testing::Message()
                             << cpu << '/' << op << '/' << mode << '/' << mask << '/' << dest);
                uint32_t words[4][34], before[4][34];
                uint32_t seed = 231;
                for (auto &reg : words) {
                  std::fill(reg, reg + 34, 0xdeadbeef);
                  for (unsigned lane = 1; lane <= 32; ++lane) {
                    seed = seed * 1664525u + 1013904223u;
                    const uint32_t special[] = {0, 1, 0x7fffffff, 0x80000000, 0xffffffff};
                    reg[lane] = lane <= 5 ? special[lane - 1] : seed;
                  }
                }
                std::memcpy(before, words, sizeof(words));
                uint32_t *p[] = {words[0] + 1, words[1] + 1, words[2] + 1, words[3] + 1};
                ASSERT_EQ(goc_test::half_fma_functions[op](cpu | sem, mask, mode, p + dest,
                                                           p + layout[0], p + layout[1],
                                                           p + layout[2]),
                          GOC_SUCCESS);
                for (unsigned reg = 0; reg < 4; ++reg)
                  for (unsigned index = 0; index < 34; ++index) {
                    uint32_t want = before[reg][index];
                    int source;
                    bool written = reg == dest && index > 0 && index < 33 &&
                                   goc_test::dpp_source(mode, mask, index - 1, source);
                    if (written)
                      want = goc_test::half_fma_result(
                          op, source < 0 ? 0 : before[layout[0]][source + 1],
                          before[layout[1]][index], before[layout[2]][index], uint32_t(mode), want);
                    if (written) {
                      ASSERT_EQ(sem ? words[reg][index]
                                    : canonical(words[reg][index], uint32_t(mode)),
                                sem ? want : canonical(want, uint32_t(mode)));
                    } else {
                      ASSERT_EQ(words[reg][index], want);
                    }
                  }
              }
}

TEST(DppHalfFma, Validation) {
  for (unsigned op = 0; op < 2; ++op)
    for (uint64_t mode : modes(op)) {
      auto fn = goc_test::half_fma_functions[op];
      EXPECT_EQ(fn(0, 0, mode, nullptr, nullptr, nullptr, nullptr), GOC_SUCCESS);
      for (uint32_t mask : {UINT32_C(0), UINT32_MAX}) {
        if (op == 1) {
          for (uint32_t invalid : {GOC_ALU_NEG_C, GOC_ALU_ABS_C, GOC_ALU_HIGH_C})
            EXPECT_EQ(fn(0, mask, mode | invalid, nullptr, nullptr, nullptr, nullptr),
                      GOC_ERROR_INVALID_FLAGS);
        }
        EXPECT_EQ(fn(0, mask, mode | (1u << 13), nullptr, nullptr, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(fn(0, mask, GOC_DPP8 | GOC_DPP16, nullptr, nullptr, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(fn(UINT64_C(1) << 63, mask, mode, nullptr, nullptr, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(fn((UINT64_C(2) << 16) | GOC_SEMANTICS_STRICT, mask, mode, nullptr, nullptr,
                     nullptr, nullptr),
                  GOC_ERROR_UNSUPPORTED_SEMANTICS);
      }
    }
}

TEST(DppHalfFma, EveryModifierAndOverflowMode) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (uint64_t sem : {UINT64_C(0), GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT})
      for (unsigned op = 0; op < 2; ++op)
        for (bool saturate : {false, true})
          for (unsigned variant = 0; variant < 8192; ++variant)
            for (uint64_t descriptor : goc_test::dpp_modes) {
              if (!goc_test::half_fma_valid_mode(op, variant))
                continue;
              uint64_t mode = descriptor | variant;
              uint32_t words[4][32];
              uint32_t seed = 997 + variant;
              for (auto &reg : words)
                for (auto &word : reg) {
                  seed = seed * 1664525u + 1013904223u;
                  word = seed;
                }
              uint32_t old[32];
              std::memcpy(old, words[3], sizeof(old));
              uint32_t *p[] = {words[0], words[1], words[2], words[3]};
              ASSERT_EQ(goc_test::half_fma_functions[op](cpu | sem | (saturate ? GOC_FP16_OVFL : 0),
                                                         UINT32_MAX, mode, p + 3, p, p + 1, p + 2),
                        GOC_SUCCESS);
              for (unsigned lane = 0; lane < 32; ++lane) {
                uint32_t want = old[lane];
                int source;
                if (goc_test::dpp_source(mode, UINT32_MAX, lane, source)) {
                  want = goc_test::half_fma_result(op, source < 0 ? 0 : words[0][source],
                                                   words[1][lane], words[2][lane], uint32_t(mode),
                                                   old[lane], saturate);
                }
                ASSERT_EQ(sem ? words[3][lane] : canonical(words[3][lane], uint32_t(mode)),
                          sem ? want : canonical(want, uint32_t(mode)))
                    << cpu << '/' << op << '/' << mode << '/' << lane << '/' << saturate;
              }
            }
}

TEST(DppHalfFma, ExactPreservesHostEnvironment) {
  goc_test::ScopedFpEnvironment restore;
  ASSERT_TRUE(restore.saved());

  const uint32_t values[] = {0x7c017c01, 0x3c010001, 0x80008000, 0x7bfffbff};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned op = 0; op < 2; ++op)
      for (uint64_t mode : modes(op))
        for (int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
          uint32_t words[4][32], expected[32];
          for (unsigned reg = 0; reg < 4; ++reg)
            for (unsigned lane = 0; lane < 32; ++lane)
              words[reg][lane] = values[(lane + reg) % 4];
          for (unsigned lane = 0; lane < 32; ++lane) {
            expected[lane] = words[3][lane];
            int source;
            if (goc_test::dpp_source(mode, UINT32_C(0xaaaaaaaa), lane, source))
              expected[lane] =
                  goc_test::half_fma_result(op, source < 0 ? 0 : words[0][source], words[1][lane],
                                            words[2][lane], uint32_t(mode), words[3][lane]);
          }
          uint32_t *p[] = {words[0], words[1], words[2], words[3]};
          ASSERT_EQ(std::fesetround(rounding), 0);
          ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
          ASSERT_EQ(std::feraiseexcept(FE_DIVBYZERO), 0);
          ASSERT_EQ(goc_test::half_fma_functions[op](
                        cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                        UINT32_C(0xaaaaaaaa), mode, p + 3, p, p + 1, p + 2),
                    GOC_SUCCESS);
          EXPECT_EQ(std::fegetround(), rounding);
          EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
          for (unsigned lane = 0; lane < 32; ++lane)
            ASSERT_EQ(words[3][lane], expected[lane]);
        }
}

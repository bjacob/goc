// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "dpp_reference.h"
#include "exec_masks.h"
#include "goc/goc.h"
#include "half_reference.h"
#include "half_unary_reference.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>
#include <vector>

namespace {

std::vector<uint64_t> modes() {
  std::vector<uint64_t> result;
  for (uint64_t descriptor : goc_test::dpp_modes)
    for (uint32_t low : {0u, 1u, 8u, 9u, 64u, 128u, 192u, 256u, 512u, 4096u, 4608u, 5065u})
      result.push_back(descriptor | low);
  return result;
}

uint32_t reference(unsigned op, uint32_t a, uint32_t old_d, uint32_t mode) {
  uint32_t value = goc_test::half_unary_reference(op, a, mode, false);
  unsigned shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  return (old_d & ~(65535U << shift)) | (value << shift);
}

void check(unsigned op, uint32_t got, uint32_t want, uint32_t mode) {
  unsigned shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  ASSERT_EQ((got ^ want) & ~(65535U << shift), 0u);
  uint16_t x = uint16_t(got >> shift), y = uint16_t(want >> shift);
  if ((y & 0x7fff) > 0x7c00) {
    ASSERT_GT(x & 0x7fff, 0x7c00);
  } else if (op >= 4 && op <= 8 && (y & 0x7fff) && (y & 0x7fff) < 0x7c00) {
    ASSERT_EQ(x & 0x8000, y & 0x8000);
    ASSERT_LE(std::abs(int(x) - int(y)), 1);
  } else {
    ASSERT_EQ(x, y);
  }
}

void hardware_corpus(const uint32_t *values, uint64_t expected_hash) {
  // Gfx1201: 132 operation/modifier combinations x 7 DPP descriptors x 8
  // EXEC masks x 32 lanes, retaining the unselected destination half.
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint64_t hash = goc_test::capture_hash_seed;
    for (uint32_t exec_mask :
         {0xffffffffu, 0u, 0xaaaaaaaau, 0x55555555u, 1u, 0x80000000u, 0xffffu, 0xffff0000u})
      for (unsigned op = 0; op < 11; ++op)
        for (uint64_t mode : modes()) {
          uint32_t a[32], d[32];
          for (unsigned lane = 0; lane < 32; ++lane) {
            a[lane] = values[lane % 8];
            d[lane] = 0xdead0000u + lane;
          }
          auto pa = a, pd = d;
          ASSERT_EQ(goc_test::half_unary_functions[op](cpu, exec_mask, mode, &pd, &pa, nullptr),
                    GOC_SUCCESS);
          for (unsigned lane = 0; lane < 32; ++lane) {
            int source;
            uint32_t want = 0xdead0000u + lane;
            if (goc_test::dpp_source(mode, exec_mask, lane, source))
              want = reference(op, source < 0 ? 0 : a[source], 0xdead0000u + lane, uint32_t(mode));
            ASSERT_EQ(goc_test::canonical_half_nan(d[lane], uint32_t(mode)),
                      goc_test::canonical_half_nan(want, uint32_t(mode)))
                << cpu << '/' << op << '/' << mode << '/' << lane;
            hash = goc_test::capture_hash_word(
                hash, goc_test::canonical_half_nan(d[lane], uint32_t(mode)));
          }
        }
    EXPECT_EQ(hash, expected_hash);
  }
}

} // namespace

TEST(DppHalfUnary, HardwareCorpus) {
  const uint32_t values[] = {0x3c00bc00u, 0xc0004000u, 0x38003400u, 0xb800b400u,
                             0x44004200u, 0xc400c200u, 0x3a003600u, 0xba00b600u};
  hardware_corpus(values, 0x3833b76d19aa40f7ULL);
}

TEST(DppHalfUnary, HardwareOmodBoundaries) {
  const uint32_t values[] = {0x00018001u, 0x03ff83ffu, 0x04008400u, 0x04018401u,
                             0x08008800u, 0x3c00bc00u, 0x00008000u, 0x7bfffbffu};
  hardware_corpus(values, 0xfa080d7a883a740dULL);
}

TEST(DppHalfUnary, EveryModifierAndOverflowMode) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned op = 0; op < 11; ++op)
      for (bool saturate : {false, true})
        for (unsigned variant = 0; variant < 128; ++variant)
          for (uint64_t descriptor : goc_test::dpp_modes) {
            uint64_t mode = descriptor | goc_test::half_unary_modifiers(variant);
            SCOPED_TRACE(::testing::Message()
                         << cpu << "/" << op << "/" << mode << "/" << saturate);
            uint32_t words[3][32];
            uint32_t seed = 997 + variant;
            for (auto &reg : words)
              for (auto &word : reg) {
                seed = seed * 1664525u + 1013904223u;
                word = seed;
              }
            uint32_t old[32];
            std::memcpy(old, words[2], sizeof(old));
            uint32_t *p[] = {words[0], words[1], words[2]};
            ASSERT_EQ(goc_test::half_unary_functions[op](cpu | (saturate ? GOC_FP16_OVFL : 0),
                                                         UINT32_MAX, mode, p + 2, p, nullptr),
                      GOC_SUCCESS);
            for (unsigned lane = 0; lane < 32; ++lane) {
              uint32_t want = old[lane];
              int source;
              if (goc_test::dpp_source(mode, UINT32_MAX, lane, source)) {
                unsigned shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
                uint32_t value = goc_test::half_unary_reference(
                    op, source < 0 ? 0 : words[0][source], uint32_t(mode), saturate);
                want = (want & ~(65535U << shift)) | (value << shift);
              }
              check(op, words[2][lane], want, uint32_t(mode));
            }
          }
}

TEST(DppHalfUnary, MasksAliasesAndRandomWords) {
  const unsigned aliases[][2] = {{1, 0}, {0, 0}};
  auto masks = exec_masks();
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned op = 0; op < 11; ++op)
      for (uint64_t mode : modes())
        for (uint32_t exec_mask : masks)
          for (const auto &alias : aliases)
            for (unsigned batch = 0; batch < 4; ++batch) {
              SCOPED_TRACE(::testing::Message()
                           << cpu << '/' << op << '/' << mode << '/' << exec_mask << '/' << alias[0]
                           << '/' << alias[1] << '/' << batch);
              uint32_t words[3][34], before[3][34];
              uint32_t seed = 231 + batch;
              for (auto &reg : words) {
                std::fill(reg, reg + 34, 0xdeadbeef);
                for (unsigned lane = 1; lane <= 32; ++lane) {
                  seed = seed * 1664525u + 1013904223u;
                  reg[lane] = batch == 0   ? (lane & 1 ? UINT32_MAX : 0)
                              : batch == 1 ? uint32_t(1) << (lane - 1)
                                           : seed;
                }
              }
              std::memcpy(before, words, sizeof(words));
              uint32_t *p[] = {words[0] + 1, words[1] + 1, words[2] + 1};
              ASSERT_EQ(goc_test::half_unary_functions[op](cpu, exec_mask, mode, p + alias[0],
                                                           p + alias[1], nullptr),
                        GOC_SUCCESS);
              for (unsigned reg = 0; reg < 3; ++reg)
                for (unsigned index = 0; index < 34; ++index) {
                  uint32_t want = before[reg][index];
                  int source;
                  bool written = reg == alias[0] && index > 0 && index < 33 &&
                                 goc_test::dpp_source(mode, exec_mask, index - 1, source);
                  if (written)
                    want = reference(op, source < 0 ? 0 : before[alias[1]][source + 1],
                                     before[alias[0]][index], uint32_t(mode));
                  if (written) {
                    check(op, words[reg][index], want, uint32_t(mode));
                  } else {
                    ASSERT_EQ(words[reg][index], want);
                  }
                }
            }
}

TEST(DppHalfUnary, Validation) {
  for (auto fn : goc_test::half_unary_functions)
    for (uint64_t mode : modes()) {
      EXPECT_EQ(fn(0, 0, mode, nullptr, nullptr, nullptr), GOC_SUCCESS);
      for (uint32_t exec_mask : {0U, UINT32_MAX}) {
        EXPECT_EQ(fn(0, exec_mask, mode | GOC_ALU_NEG_C, nullptr, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(fn(0, exec_mask, GOC_DPP8 | GOC_DPP16, nullptr, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(fn(1ULL << 63, exec_mask, mode, nullptr, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, exec_mask, mode, nullptr,
                     nullptr, nullptr),
                  GOC_ERROR_UNSUPPORTED_SEMANTICS);
      }
    }
}

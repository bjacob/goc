// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "goc/goc.h"
#include "rdna4_dpp_reference.h"
#include "rdna4_exec_masks.h"
#include "rdna4_half_binary_reference.h"
#include "rdna4_half_reference.h"

#include <algorithm>
#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>
#include <vector>

namespace {

std::vector<uint64_t> modes() {
  std::vector<uint64_t> result;
  for (uint64_t descriptor : goc_test::dpp_modes)
    for (unsigned selection :
         {0u, 1u, 2u, 4u, 8u, 16u, 32u, 48u, 64u, 127u, 128u, 256u, 512u, 896u, 1023u, 943u})
      result.push_back(descriptor | goc_test::half_binary_modifiers(selection));
  return result;
}

uint32_t reference(unsigned op, uint32_t a, uint32_t b, uint32_t old_d, uint32_t mode) {
  uint32_t value = goc_test::half_binary_reference(op, a, b, mode, false);
  unsigned shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  return (old_d & ~(65535U << shift)) | (value << shift);
}

void hardware_corpus(const uint32_t *values, uint64_t expected_hash) {
  // Gfx1201: 128 operation/modifier combinations x 7 DPP descriptors x 8
  // EXEC masks x 32 lanes, retaining the unselected destination half.
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint64_t hash = goc_test::capture_hash_seed;
    for (uint32_t exec_mask :
         {0xffffffffu, 0u, 0xaaaaaaaau, 0x55555555u, 1u, 0x80000000u, 0xffffu, 0xffff0000u})
      for (unsigned op = 0; op < 8; ++op)
        for (uint64_t mode : modes()) {
          uint32_t a[32], b[32], d[32];
          for (unsigned lane = 0; lane < 32; ++lane) {
            a[lane] = values[lane % 8];
            b[lane] = values[(lane + 3) % 8];
            d[lane] = 0xdead0000u + lane;
          }
          auto pa = a, pb = b, pd = d;
          ASSERT_EQ(goc_test::half_binary_functions[op](cpu, exec_mask, mode, &pd, &pa, &pb),
                    GOC_SUCCESS);
          for (unsigned lane = 0; lane < 32; ++lane) {
            int source;
            uint32_t want = 0xdead0000u + lane;
            if (goc_test::dpp_source(mode, exec_mask, lane, source))
              want = reference(op, source < 0 ? 0 : a[source], b[lane], 0xdead0000u + lane,
                               uint32_t(mode));
            ASSERT_EQ(d[lane], want) << cpu << '/' << op << '/' << mode << '/' << lane;
            hash = goc_test::capture_hash_word(hash, d[lane]);
          }
        }
    EXPECT_EQ(hash, expected_hash);
  }
}

} // namespace

TEST(DppHalfBinary, HardwareCorpus) {
  const uint32_t values[] = {0x3c00bc00u, 0xc0004000u, 0x38003400u, 0xb800b400u,
                             0x44004200u, 0xc400c200u, 0x3a003600u, 0xba00b600u};
  hardware_corpus(values, 0xd7b780a730315ee5ULL);
}

TEST(DppHalfBinary, HardwareOmodBoundaries) {
  const uint32_t values[] = {0x00018001u, 0x03ff83ffu, 0x04008400u, 0x04018401u,
                             0x08008800u, 0x3c00bc00u, 0x00008000u, 0x7bfffbffu};
  hardware_corpus(values, 0x306a62849f129dbULL);
}

TEST(DppHalfBinary, EveryModifierAndOverflowMode) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned op = 0; op < 8; ++op)
      for (bool saturate : {false, true})
        for (unsigned variant = 0; variant < 1024; ++variant)
          for (uint64_t descriptor : goc_test::dpp_modes) {
            uint64_t mode = descriptor | goc_test::half_binary_modifiers(variant);
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
            ASSERT_EQ(goc_test::half_binary_functions[op](cpu | (saturate ? GOC_FP16_OVFL : 0),
                                                          UINT32_MAX, mode, p + 2, p, p + 1),
                      GOC_SUCCESS);
            for (unsigned lane = 0; lane < 32; ++lane) {
              uint32_t want = old[lane];
              int source;
              if (goc_test::dpp_source(mode, UINT32_MAX, lane, source)) {
                unsigned shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
                uint32_t value =
                    goc_test::half_binary_reference(op, source < 0 ? 0 : words[0][source],
                                                    words[1][lane], uint32_t(mode), saturate);
                want = (want & ~(65535U << shift)) | (value << shift);
              }
              ASSERT_EQ(goc_test::canonical_half_nan(words[2][lane], uint32_t(mode)),
                        goc_test::canonical_half_nan(want, uint32_t(mode)))
                  << cpu << '/' << op << '/' << mode << '/' << lane << '/' << saturate;
            }
          }
}

TEST(DppHalfBinary, MasksAliasesAndRandomWords) {
  const unsigned aliases[][3] = {{2, 0, 1}, {0, 0, 1}, {1, 0, 1}, {0, 0, 0}, {2, 0, 0}};
  auto masks = rdna4_exec_masks();
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned op = 0; op < 8; ++op)
      for (uint64_t mode : modes())
        for (uint32_t exec_mask : masks)
          for (const auto &alias : aliases)
            for (unsigned batch = 0; batch < 4; ++batch) {
              SCOPED_TRACE(::testing::Message()
                           << cpu << '/' << op << '/' << mode << '/' << exec_mask << '/' << alias[0]
                           << '/' << alias[2] << '/' << batch);
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
              ASSERT_EQ(goc_test::half_binary_functions[op](cpu, exec_mask, mode, p + alias[0],
                                                            p + alias[1], p + alias[2]),
                        GOC_SUCCESS);
              for (unsigned reg = 0; reg < 3; ++reg)
                for (unsigned index = 0; index < 34; ++index) {
                  uint32_t want = before[reg][index];
                  int source;
                  bool written = reg == alias[0] && index > 0 && index < 33 &&
                                 goc_test::dpp_source(mode, exec_mask, index - 1, source);
                  if (written)
                    want =
                        reference(op, source < 0 ? 0 : before[alias[1]][source + 1],
                                  before[alias[2]][index], before[alias[0]][index], uint32_t(mode));
                  if (written) {
                    ASSERT_EQ(goc_test::canonical_half_nan(words[reg][index], uint32_t(mode)),
                              goc_test::canonical_half_nan(want, uint32_t(mode)));
                  } else {
                    ASSERT_EQ(words[reg][index], want);
                  }
                }
            }
}

TEST(DppHalfBinary, Validation) {
  for (auto fn : goc_test::half_binary_functions)
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

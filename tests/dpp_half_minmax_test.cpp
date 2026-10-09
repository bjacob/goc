// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "dpp_reference.h"
#include "exec_masks.h"
#include "goc/goc.h"
#include "half_minmax_reference.h"
#include "half_reference.h"

#include <algorithm>
#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>
#include <vector>

namespace {

std::vector<uint64_t> modes() {
  std::vector<uint64_t> result;
  for (uint64_t descriptor : goc_test::dpp_modes)
    for (unsigned low : {0u, 1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u, 192u, 256u, 512u, 1024u, 2048u,
                         4096u, 7680u, 8191u, 7935u})
      result.push_back(descriptor | low);
  return result;
}

} // namespace

namespace {

uint32_t reference(unsigned op, uint32_t a, uint32_t b, uint32_t c, uint32_t mode, uint32_t old_d) {
  uint32_t value = goc_test::half_minmax_reference(op, a, b, c, mode, false);
  unsigned shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  return (old_d & ~(65535U << shift)) | (value << shift);
}

void hardware_corpus(const uint32_t *values, uint64_t expected_hash) {
  // Gfx1201: nine operations x 18 modifier combinations x seven DPP
  // descriptors x eight EXEC masks x 32 lanes.
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint64_t hash = goc_test::capture_hash_seed;
    for (uint32_t exec_mask :
         {0xffffffffu, 0u, 0xaaaaaaaau, 0x55555555u, 1u, 0x80000000u, 0xffffu, 0xffff0000u})
      for (unsigned op = 0; op < 9; ++op)
        for (uint64_t mode : modes()) {
          uint32_t a[32], b[32], c[32], d[32];
          for (unsigned lane = 0; lane < 32; ++lane) {
            a[lane] = values[lane % 8];
            b[lane] = values[(lane + 3) % 8];
            c[lane] = values[(lane + 5) % 8];
            d[lane] = 0xdead0000u + lane;
          }
          auto pa = a, pb = b, pc = c, pd = d;
          ASSERT_EQ(goc_test::half_minmax_functions[op](cpu, exec_mask, mode, &pd, &pa, &pb, &pc,
                                                        nullptr),
                    GOC_SUCCESS);
          for (unsigned lane = 0; lane < 32; ++lane) {
            int source;
            uint32_t want = 0xdead0000u + lane;
            if (goc_test::dpp_source(mode, exec_mask, lane, source))
              want =
                  reference(op, source < 0 ? 0 : a[source], b[lane], c[lane], uint32_t(mode), want);
            ASSERT_EQ(d[lane], want) << cpu << '/' << op << '/' << mode << '/' << lane;
            hash = goc_test::capture_hash_word(hash, d[lane]);
          }
        }
    EXPECT_EQ(hash, expected_hash);
  }
}

} // namespace

TEST(DppHalfMinmax, HardwareCorpus) {
  const uint32_t values[] = {0x3c00bc00u, 0xc0004000u, 0x38003400u, 0xb800b400u,
                             0x44004200u, 0xc400c200u, 0x3a003600u, 0xba00b600u};
  hardware_corpus(values, 0xb0dcac9d15c8cc85ULL);
}

TEST(DppHalfMinmax, HardwareOmodBoundaries) {
  const uint32_t values[] = {0x00018001u, 0x03ff83ffu, 0x04008400u, 0x04018401u,
                             0x08008800u, 0x3c00bc00u, 0x00008000u, 0x7bfffbffu};
  hardware_corpus(values, 0xc587f06661dd8f15ULL);
}

TEST(DppHalfMinmax, MasksAliasesAndRandomWords) {
  const unsigned layouts[][3] = {{0, 1, 2}, {0, 0, 2}, {0, 1, 0}, {0, 1, 1}, {0, 0, 0}};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned op = 0; op < 9; ++op)
      for (uint64_t mode : modes())
        for (uint32_t exec_mask : exec_masks())
          for (const auto &layout : layouts)
            for (unsigned dest = 0; dest < 4; ++dest) {
              SCOPED_TRACE(::testing::Message()
                           << cpu << '/' << op << '/' << mode << '/' << exec_mask << '/' << dest);
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
              ASSERT_EQ(goc_test::half_minmax_functions[op](cpu, exec_mask, mode, p + dest,
                                                            p + layout[0], p + layout[1],
                                                            p + layout[2], nullptr),
                        GOC_SUCCESS);
              for (unsigned reg = 0; reg < 4; ++reg)
                for (unsigned index = 0; index < 34; ++index) {
                  uint32_t want = before[reg][index];
                  int source;
                  bool written = reg == dest && index > 0 && index < 33 &&
                                 goc_test::dpp_source(mode, exec_mask, index - 1, source);
                  if (written)
                    want = reference(op, source < 0 ? 0 : before[layout[0]][source + 1],
                                     before[layout[1]][index], before[layout[2]][index],
                                     uint32_t(mode), want);
                  if (written) {
                    ASSERT_EQ(goc_test::canonical_half_nan(words[reg][index], uint32_t(mode)),
                              goc_test::canonical_half_nan(want, uint32_t(mode)));
                  } else {
                    ASSERT_EQ(words[reg][index], want);
                  }
                }
            }
}

TEST(DppHalfMinmax, Validation) {
  for (auto fn : goc_test::half_minmax_functions)
    for (uint64_t mode : modes()) {
      EXPECT_EQ(fn(0, 0, mode, nullptr, nullptr, nullptr, nullptr, nullptr), GOC_SUCCESS);
      for (uint32_t exec_mask : {0U, UINT32_MAX}) {
        EXPECT_EQ(fn(0, exec_mask, mode | (1u << 13), nullptr, nullptr, nullptr, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(
            fn(0, exec_mask, GOC_DPP8 | GOC_DPP16, nullptr, nullptr, nullptr, nullptr, nullptr),
            GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(fn(1ULL << 63, exec_mask, mode, nullptr, nullptr, nullptr, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, exec_mask, mode, nullptr,
                     nullptr, nullptr, nullptr, nullptr),
                  GOC_ERROR_UNSUPPORTED_SEMANTICS);
      }
    }
}

TEST(DppHalfMinmax, EveryModifierAndOverflowMode) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned op = 0; op < 9; ++op)
      for (bool saturate : {false, true})
        for (unsigned variant = 0; variant < 8192; ++variant)
          for (uint64_t descriptor : goc_test::dpp_modes) {
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
            ASSERT_EQ(goc_test::half_minmax_functions[op](cpu | (saturate ? GOC_FP16_OVFL : 0),
                                                          UINT32_MAX, mode, p + 3, p, p + 1, p + 2,
                                                          nullptr),
                      GOC_SUCCESS);
            for (unsigned lane = 0; lane < 32; ++lane) {
              uint32_t want = old[lane];
              int source;
              if (goc_test::dpp_source(mode, UINT32_MAX, lane, source)) {
                unsigned shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
                uint32_t value = goc_test::half_minmax_reference(
                    op, source < 0 ? 0 : words[0][source], words[1][lane], words[2][lane],
                    uint32_t(mode), saturate);
                want = (want & ~(65535U << shift)) | (value << shift);
              }
              ASSERT_EQ(goc_test::canonical_half_nan(words[3][lane], uint32_t(mode)),
                        goc_test::canonical_half_nan(want, uint32_t(mode)))
                  << cpu << '/' << op << '/' << mode << '/' << lane << '/' << saturate;
            }
          }
}

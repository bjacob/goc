// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "dpp_reference.h"
#include "exec_masks.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "integer_mad_reference.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>
#include <vector>

namespace {

std::vector<uint64_t> modes(unsigned op) {
  std::vector<uint64_t> result;
  for (uint64_t descriptor : goc_test::dpp_modes)
    for (unsigned low = 0; low < (op < 2 ? 8u : 2u); ++low)
      result.push_back(descriptor | goc_test::integer_mad_mode_bits(low));
  return result;
}

} // namespace

TEST(DppIntegerMad, HardwareCorpus) {
  // Gfx1201: 20 operation/CLAMP/selector combinations x 7 DPP descriptors
  // x 8 EXEC masks x 32 lanes, signed/unsigned 16-bit and 24-bit MAD.
  const uint32_t values[] = {0,           0xffffffffu, 1,           0x80000000u,
                             0x7fffffffu, 0xaaaaaaaau, 0x55555555u, 0x01010101u};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint64_t hash = goc_test::capture_hash_seed;
    for (uint32_t exec_mask :
         {0xffffffffu, 0u, 0xaaaaaaaau, 0x55555555u, 1u, 0x80000000u, 0xffffu, 0xffff0000u})
      for (unsigned op = 0; op < 4; ++op)
        for (uint64_t mode : modes(op)) {
          uint32_t a[32], b[32], c[32], d[32];
          for (unsigned lane = 0; lane < 32; ++lane) {
            a[lane] = values[lane % 8] ^ ((lane / 8) * 0x01010101u);
            b[lane] = 0xffffff00u + lane;
            c[lane] = values[(lane + 3) % 8] ^ ((lane / 8) * 0x10101010u);
            d[lane] = 0xdead0000u + lane;
          }
          auto pa = a, pb = b, pc = c, pd = d;
          ASSERT_EQ(goc_test::integer_mad_functions[op](cpu, exec_mask, mode, &pd, &pa, &pb, &pc),
                    GOC_SUCCESS);
          for (unsigned lane = 0; lane < 32; ++lane) {
            int source;
            uint32_t want = 0xdead0000u + lane;
            if (goc_test::dpp_source(mode, exec_mask, lane, source))
              want = goc_test::integer_mad_reference(op, source < 0 ? 0 : a[source], b[lane],
                                                     c[lane], uint32_t(mode));
            ASSERT_EQ(d[lane], want) << cpu << '/' << op << '/' << mode << '/' << lane;
            hash = goc_test::capture_hash_word(hash, d[lane]);
          }
        }
    EXPECT_EQ(hash, 0xa596990eb402586cULL);
  }
}

TEST(DppIntegerMad, MasksAliasesAndRandomWords) {
  const unsigned layouts[][3] = {{0, 1, 2}, {0, 0, 2}, {0, 1, 0}, {0, 1, 1}, {0, 0, 0}};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned op = 0; op < 4; ++op)
      for (uint64_t mode : modes(op))
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
              ASSERT_EQ(goc_test::integer_mad_functions[op](cpu, exec_mask, mode, p + dest,
                                                            p + layout[0], p + layout[1],
                                                            p + layout[2]),
                        GOC_SUCCESS);
              for (unsigned reg = 0; reg < 4; ++reg)
                for (unsigned index = 0; index < 34; ++index) {
                  uint32_t want = before[reg][index];
                  int source;
                  if (reg == dest && index > 0 && index < 33 &&
                      goc_test::dpp_source(mode, exec_mask, index - 1, source))
                    want = goc_test::integer_mad_reference(
                        op, source < 0 ? 0 : before[layout[0]][source + 1],
                        before[layout[1]][index], before[layout[2]][index], uint32_t(mode));
                  ASSERT_EQ(words[reg][index], want);
                }
            }
}

TEST(DppIntegerMad, ValidationAndHostFpState) {
  goc_test::ScopedFpEnvironment restore;
  ASSERT_TRUE(restore.saved());

  for (unsigned op = 0; op < 4; ++op)
    for (uint64_t mode : modes(op)) {
      auto fn = goc_test::integer_mad_functions[op];
      for (uint32_t exec_mask : {0U, UINT32_MAX}) {
        if (op >= 2) {
          EXPECT_EQ(fn(0, exec_mask, mode | GOC_ALU_HIGH_A, nullptr, nullptr, nullptr, nullptr),
                    GOC_ERROR_INVALID_FLAGS);
          EXPECT_EQ(fn(0, exec_mask, mode | GOC_ALU_HIGH_B, nullptr, nullptr, nullptr, nullptr),
                    GOC_ERROR_INVALID_FLAGS);
        }
        EXPECT_EQ(fn(0, exec_mask, mode | GOC_ALU_NEG_A, nullptr, nullptr, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(fn(1ULL << 63, exec_mask, mode, nullptr, nullptr, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, exec_mask, mode, nullptr,
                     nullptr, nullptr, nullptr),
                  GOC_ERROR_UNSUPPORTED_SEMANTICS);
        EXPECT_EQ(fn(0, exec_mask, GOC_DPP8 | GOC_DPP16, nullptr, nullptr, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
      }
      EXPECT_EQ(fn(0, 0, mode, nullptr, nullptr, nullptr, nullptr), GOC_SUCCESS);
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
          uint32_t a[32], b[32], c[32], d[32];
          std::fill(a, a + 32, UINT32_MAX);
          std::fill(b, b + 32, UINT32_MAX);
          std::fill(c, c + 32, 0x80000000u);
          auto pa = a, pb = b, pc = c, pd = d;
          ASSERT_EQ(std::fesetround(rounding), 0);
          ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
          ASSERT_EQ(std::feraiseexcept(FE_DIVBYZERO), 0);
          ASSERT_EQ(fn(cpu, UINT32_MAX, mode, &pd, &pa, &pb, &pc), GOC_SUCCESS);
          EXPECT_EQ(std::fegetround(), rounding);
          EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
        }
    }
}

// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "rdna4_dpp_reference.h"
#include "rdna4_exec_masks.h"
#include "rdna4_integer_mul_reference.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>
#include <vector>

namespace {

std::vector<uint64_t> modes(unsigned op) {
  std::vector<uint64_t> result;
  for (uint64_t descriptor : goc_test::dpp_modes) {
    result.push_back(descriptor);
    if (goc_test::integer_mul_can_clamp(op))
      result.push_back(descriptor | GOC_ALU_CLAMP);
  }
  return result;
}

} // namespace

TEST(DppIntegerMul, HardwareCorpus) {
  // Gfx1201: 6 operation/CLAMP combinations x 7 DPP descriptors x 8
  // EXEC masks x 32 lanes, signed/unsigned 24-bit low/high multiplication.
  const uint32_t values[] = {0,           0xffffffffu, 1,           0x80000000u,
                             0x7fffffffu, 0xaaaaaaaau, 0x55555555u, 0x01010101u};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint64_t hash = goc_test::capture_hash_seed;
    for (uint32_t mask :
         {0xffffffffu, 0u, 0xaaaaaaaau, 0x55555555u, 1u, 0x80000000u, 0xffffu, 0xffff0000u})
      for (unsigned op = 3; op < 7; ++op)
        for (uint64_t mode : modes(op)) {
          uint32_t a[32], b[32], d[32];
          for (unsigned lane = 0; lane < 32; ++lane) {
            a[lane] = values[lane % 8] ^ ((lane / 8) * 0x01010101u);
            b[lane] = 0xffffff00u + lane;
            d[lane] = 0xdead0000u + lane;
          }
          auto pa = a, pb = b, pd = d;
          ASSERT_EQ(goc_test::integer_mul_functions[op](cpu, mask, mode, &pd, &pa, &pb),
                    GOC_SUCCESS);
          for (unsigned lane = 0; lane < 32; ++lane) {
            int source;
            uint32_t want = 0xdead0000u + lane;
            if (goc_test::dpp_source(mode, mask, lane, source))
              want = goc_test::integer_mul_reference(op, source < 0 ? 0 : a[source], b[lane],
                                                     mode & GOC_ALU_CLAMP);
            ASSERT_EQ(d[lane], want) << cpu << '/' << op << '/' << mode << '/' << lane;
            hash = goc_test::capture_hash_word(hash, d[lane]);
          }
        }
    EXPECT_EQ(hash, UINT64_C(0x9fcb727d0dced850));
  }
}

TEST(DppIntegerMul, MasksAliasesAndRandomWords) {
  const unsigned aliases[][3] = {{2, 0, 1}, {0, 0, 1}, {1, 0, 1}, {0, 0, 0}, {2, 0, 0}};
  auto masks = rdna4_exec_masks();
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned op = 3; op < 7; ++op)
      for (uint64_t mode : modes(op))
        for (uint32_t mask : masks)
          for (const auto &alias : aliases)
            for (unsigned batch = 0; batch < 8; ++batch) {
              SCOPED_TRACE(::testing::Message()
                           << cpu << '/' << op << '/' << mode << '/' << mask << '/' << alias[0]
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
              ASSERT_EQ(goc_test::integer_mul_functions[op](cpu, mask, mode, p + alias[0],
                                                            p + alias[1], p + alias[2]),
                        GOC_SUCCESS);
              for (unsigned reg = 0; reg < 3; ++reg)
                for (unsigned index = 0; index < 34; ++index) {
                  uint32_t want = before[reg][index];
                  int source;
                  if (reg == alias[0] && index > 0 && index < 33 &&
                      goc_test::dpp_source(mode, mask, index - 1, source))
                    want = goc_test::integer_mul_reference(
                        op, source < 0 ? 0 : before[alias[1]][source + 1], before[alias[2]][index],
                        mode & GOC_ALU_CLAMP);
                  ASSERT_EQ(words[reg][index], want);
                }
            }
}

TEST(DppIntegerMul, ValidationAndHostFpState) {
  goc_test::ScopedFpEnvironment restore;
  ASSERT_TRUE(restore.saved());

  for (unsigned op = 3; op < 7; ++op)
    for (uint64_t mode : modes(op)) {
      auto fn = goc_test::integer_mul_functions[op];
      for (uint32_t mask : {UINT32_C(0), UINT32_MAX}) {
        if (!goc_test::integer_mul_can_clamp(op)) {
          EXPECT_EQ(fn(0, mask, mode | GOC_ALU_CLAMP, nullptr, nullptr, nullptr),
                    GOC_ERROR_INVALID_FLAGS);
        }
        EXPECT_EQ(fn(0, mask, mode | GOC_ALU_NEG_A, nullptr, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(fn(UINT64_C(1) << 63, mask, mode, nullptr, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, mask, mode, nullptr,
                     nullptr, nullptr),
                  GOC_ERROR_UNSUPPORTED_SEMANTICS);
        EXPECT_EQ(fn(0, mask, GOC_DPP8 | GOC_DPP16, nullptr, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
      }
      EXPECT_EQ(fn(0, 0, mode, nullptr, nullptr, nullptr), GOC_SUCCESS);
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
          uint32_t a[32], b[32], d[32];
          std::fill(a, a + 32, UINT32_MAX);
          std::fill(b, b + 32, UINT32_MAX);
          auto pa = a, pb = b, pd = d;
          ASSERT_EQ(std::fesetround(rounding), 0);
          ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
          ASSERT_EQ(std::feraiseexcept(FE_DIVBYZERO), 0);
          ASSERT_EQ(fn(cpu, UINT32_MAX, mode, &pd, &pa, &pb), GOC_SUCCESS);
          EXPECT_EQ(std::fegetround(), rounding);
          EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
        }
    }
}

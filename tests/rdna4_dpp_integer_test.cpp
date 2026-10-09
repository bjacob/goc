// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_dpp_integer_reference.h"
#include "rdna4_dpp_reference.h"
#include "rdna4_exec_masks.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>

TEST(DppInteger, HardwareCorpus) {
  // Gfx1201: 11 operations x 7 DPP/FI/BOUND/row/bank descriptors x 8
  // EXEC masks x 32 lanes, including destination-lane MBCNT counting.
  const uint32_t values[] = {0,           0xffffffffu, 1,           0x80000000u,
                             0x7fffffffu, 0xaaaaaaaau, 0x55555555u, 0x01010101u};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (uint32_t mask :
         {0xffffffffu, 0u, 0xaaaaaaaau, 0x55555555u, 1u, 0x80000000u, 0xffffu, 0xffff0000u})
      for (unsigned op = 0; op < 11; ++op)
        for (uint64_t mode : goc_test::dpp_modes) {
          uint32_t a[32], b[32], d[32];
          for (unsigned lane = 0; lane < 32; ++lane) {
            a[lane] = values[lane % 8] ^ ((lane / 8) * 0x01010101u);
            b[lane] = 0xffffff00u + lane;
            d[lane] = 0xdead0000u + lane;
          }
          auto pa = a, pb = b, pd = d;
          ASSERT_EQ(goc_test::dpp_integer_functions[op](cpu, mask, mode, &pd, &pa, &pb),
                    GOC_SUCCESS);
          for (unsigned lane = 0; lane < 32; ++lane) {
            int source;
            uint32_t want = 0xdead0000u + lane;
            if (goc_test::dpp_source(mode, mask, lane, source))
              want = goc_test::dpp_integer_reference(op, source < 0 ? 0 : a[source], b[lane], lane);
            ASSERT_EQ(d[lane], want) << cpu << '/' << op << '/' << mode << '/' << lane;
            hash = (hash ^ d[lane]) * UINT64_C(1099511628211);
          }
        }
    EXPECT_EQ(hash, UINT64_C(0x5da3e13cc67cda25));
  }
}

TEST(DppInteger, MasksAliasesAndRandomWords) {
  const unsigned aliases[][3] = {{2, 0, 1}, {0, 0, 1}, {1, 0, 1}, {0, 0, 0}, {2, 0, 0}};
  auto masks = rdna4_exec_masks();
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned op = 0; op < 11; ++op)
      for (uint64_t mode : goc_test::dpp_modes)
        for (uint64_t mask : masks)
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
              ASSERT_EQ(goc_test::dpp_integer_functions[op](cpu, mask, mode, p + alias[0],
                                                            p + alias[1], p + alias[2]),
                        GOC_SUCCESS);
              for (unsigned reg = 0; reg < 3; ++reg)
                for (unsigned index = 0; index < 34; ++index) {
                  uint32_t want = before[reg][index];
                  int source;
                  if (reg == alias[0] && index > 0 && index < 33 &&
                      goc_test::dpp_source(mode, uint32_t(mask), index - 1, source))
                    want = goc_test::dpp_integer_reference(
                        op, source < 0 ? 0 : before[alias[1]][source + 1], before[alias[2]][index],
                        index - 1);
                  ASSERT_EQ(words[reg][index], want);
                }
            }
}

TEST(DppInteger, ValidationAndHostFpState) {
  struct Restore {
    std::fenv_t saved;

    Restore() { std::fegetenv(&saved); }

    ~Restore() { std::fesetenv(&saved); }
  } restore;

  for (auto fn : goc_test::dpp_integer_functions)
    for (uint64_t mode : goc_test::dpp_modes) {
      for (uint64_t mask : {UINT64_C(0), UINT64_MAX}) {
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

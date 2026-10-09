// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "rdna4_exec_masks.h"
#include "rdna4_permlane_reference.h"

#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

TEST(Permlane, HardwareCorpus) {
  // GFX1201 / HIP 7.13: 4 operations, 4 mode combinations, 16 selector
  // patterns, 8 EXEC masks, and all 32 destination lanes (65536 raw results).
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint64_t hash = goc_test::capture_hash_seed;
    for (uint32_t exec_mask :
         {0xffffffffu, 0u, 0xaaaaaaaau, 0x55555555u, 1u, 0x80000000u, 0xffffu, 0xffff0000u})
      for (unsigned op = 0; op < 4; ++op)
        for (unsigned mode = 0; mode < 4; ++mode)
          for (unsigned pattern = 0; pattern < 16; ++pattern) {
            uint32_t a[32], b[32], d[32], lo = 0, hi = 0;
            for (unsigned j = 0; j < 8; ++j) {
              lo |= ((j * 7 + pattern) & 15) << (j * 4);
              hi |= (((j + 8) * 7 + pattern) & 15) << (j * 4);
            }
            for (unsigned lane = 0; lane < 32; ++lane) {
              a[lane] = 0xabc00000u + lane;
              b[lane] = (lane * 7 + pattern) & 15;
              d[lane] = 0xdef00000u + lane;
            }
            uint32_t *pd = d;
            const uint32_t *pa = a, *pb = b;
            ASSERT_EQ(goc_test::permlane_call(
                          op, cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, exec_mask,
                          mode, &pd, &pa, &pb, lo, hi),
                      GOC_SUCCESS);
            for (unsigned lane = 0; lane < 32; ++lane) {
              ASSERT_EQ(d[lane], goc_test::permlane_reference(op, exec_mask, mode, lane, a, b,
                                                              0xdef00000u + lane, lo, hi));
              hash = goc_test::capture_hash_word(hash, d[lane]);
            }
          }
    EXPECT_EQ(hash, 0xb98b97330147d825ULL);
  }
}

TEST(Permlane, MasksAliasingAndSelectors) {
  std::mt19937 rng(72815);
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
      for (unsigned op = 0; op < 4; ++op)
        for (unsigned mode = 0; mode < 4; ++mode)
          for (uint32_t exec_mask : rdna4_exec_masks())
            for (unsigned alias = 0; alias < 4; ++alias) {
              uint32_t storage[3][34], before[3][34], want[32];
              for (auto &reg : storage)
                for (auto &word : reg)
                  word = rng();
              std::memcpy(before, storage, sizeof(storage));
              unsigned dest = alias == 0 ? 2 : alias == 2 ? 1 : 0;
              unsigned bi = alias == 3 ? 0 : 1;
              uint32_t lo = rng(), hi = rng();
              for (unsigned lane = 0; lane < 32; ++lane)
                want[lane] =
                    goc_test::permlane_reference(op, exec_mask, mode, lane, before[0] + 1,
                                                 before[bi] + 1, before[dest][lane + 1], lo, hi);
              uint32_t *pd = storage[dest] + 1;
              const uint32_t *pa = storage[0] + 1, *pb = storage[bi] + 1;
              ASSERT_EQ(goc_test::permlane_call(op, cpu | semantics | GOC_SEMANTICS_STRICT,
                                                exec_mask, mode, &pd, &pa, &pb, lo, hi),
                        GOC_SUCCESS);
              for (unsigned reg = 0; reg < 3; ++reg)
                for (unsigned lane = 0; lane < 34; ++lane)
                  ASSERT_EQ(storage[reg][lane], reg == dest && lane > 0 && lane < 33
                                                    ? want[lane - 1]
                                                    : before[reg][lane]);
            }
}

TEST(Permlane, FlagsAndHostState) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    ASSERT_EQ(std::fesetround(rounding), 0);
    std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_DIVBYZERO);
    int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned op = 0; op < 4; ++op) {
        uint32_t a[32] = {}, d[32] = {};
        uint32_t *pd = d;
        const uint32_t *pa = a;
        for (unsigned mode = 0; mode < 4; ++mode)
          ASSERT_EQ(goc_test::permlane_call(op,
                                            cpu | GOC_FP_FLUSH_INPUT_DENORMALS |
                                                GOC_FP_FLUSH_OUTPUT_DENORMALS | GOC_FP16_OVFL,
                                            UINT32_MAX, mode, &pd, &pa, &pa, 0, 0),
                    GOC_SUCCESS);
        d[0] = 123;
        for (unsigned bit = 2; bit < 32; ++bit)
          EXPECT_EQ(goc_test::permlane_call(op, cpu, 0, 1u << bit, &pd, &pa, &pa, 0, 0),
                    GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(goc_test::permlane_call(op, 1ULL << 63, 0, 0, &pd, &pa, &pa, 0, 0),
                  GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(d[0], 123u);
        EXPECT_EQ(std::fegetround(), rounding);
        EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
      }
  }
}

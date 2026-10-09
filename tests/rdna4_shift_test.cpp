// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_exec_masks.h"
#include "rdna4_shift_reference.h"

#include <algorithm>
#include <cfenv>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace {

using Fn = decltype(&goc_rdna4_v_lshlrev_b32);
const Fn functions[] = {goc_rdna4_v_lshlrev_b32, goc_rdna4_v_lshrrev_b32, goc_rdna4_v_ashrrev_i32,
                        goc_rdna4_v_lshlrev_b64, goc_rdna4_v_lshrrev_b64, goc_rdna4_v_ashrrev_i64};

} // namespace

TEST(Shift, EveryCountAndBitPosition) {
  for (int op = 0; op < 6; ++op) {
    int bits = op < 3 ? 32 : 64;
    uint64_t mask = bits == 32 ? UINT32_MAX : UINT64_MAX;
    std::vector<uint64_t> values = {0, mask, UINT64_C(0xaaaaaaaaaaaaaaaa) & mask,
                                    UINT64_C(0x5555555555555555) & mask};
    for (int bit = 0; bit < bits; ++bit) {
      values.push_back(UINT64_C(1) << bit);
      values.push_back(mask ^ (UINT64_C(1) << bit));
    }
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint64_t value : values)
        for (uint32_t start = 0; start < unsigned(bits * 4); start += 32) {
          uint32_t a[32], b[2][32], d[2][32];
          const uint32_t *ap[] = {a}, *bp[] = {b[0], b[1]};
          uint32_t *dp[] = {d[0], d[1]};
          for (uint32_t lane = 0; lane < 32; ++lane) {
            a[lane] = start + lane;
            b[0][lane] = uint32_t(value);
            b[1][lane] = uint32_t(value >> 32);
            d[0][lane] = d[1][lane] = 0xdeadbeef;
          }
          ASSERT_EQ(functions[op](cpu, UINT32_MAX, 0, dp, ap, bp), GOC_SUCCESS);
          for (int lane = 0; lane < 32; ++lane) {
            uint64_t expected = goc_test::shift_reference(bits, op % 3, a[lane], value);
            ASSERT_EQ(d[0][lane], uint32_t(expected)) << op << "/" << cpu << "/" << a[lane];
            ASSERT_EQ(d[1][lane], bits == 64 ? uint32_t(expected >> 32) : 0xdeadbeef);
          }
        }
  }
}

TEST(Shift, RandomOperandsAndIgnoredCountBits) {
  for (int op = 0; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      const int bits = op < 3 ? 32 : 64;
      std::mt19937 random(4401);
      for (int start = 0; start < 8192; start += 32) {
        uint32_t a[32], reduced[32], b[2][32], d[2][32], shortened[2][32];
        const uint32_t *ap[] = {a}, *rp[] = {reduced}, *bp[] = {b[0], b[1]};
        uint32_t *dp[] = {d[0], d[1]}, *sp[] = {shortened[0], shortened[1]};
        for (int lane = 0; lane < 32; ++lane) {
          a[lane] = random();
          reduced[lane] = a[lane] & (bits - 1);
          b[0][lane] = random();
          b[1][lane] = random();
        }
        ASSERT_EQ(functions[op](cpu, UINT32_MAX, 0, dp, ap, bp), GOC_SUCCESS);
        ASSERT_EQ(functions[op](cpu, UINT32_MAX, 0, sp, rp, bp), GOC_SUCCESS);
        for (int lane = 0; lane < 32; ++lane) {
          const uint64_t value = b[0][lane] | (uint64_t(b[1][lane]) << 32);
          const uint64_t expected = goc_test::shift_reference(bits, op % 3, a[lane], value);
          ASSERT_EQ(d[0][lane], uint32_t(expected));
          ASSERT_EQ(d[0][lane], shortened[0][lane]);
          if (bits == 64) {
            ASSERT_EQ(d[1][lane], uint32_t(expected >> 32));
            ASSERT_EQ(d[1][lane], shortened[1][lane]);
          }
        }
      }
    }
}

TEST(Shift, MasksAndEveryDestinationAlias) {
  const int sources[][3] = {{0, 1, 2}, {1, 1, 2}, {2, 1, 2}, {0, 1, 1}, {1, 1, 1}};
  for (int op = 0; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (const auto &source : sources) {
        const int bits = op < 3 ? 32 : 64;
        std::mt19937 random(324);
        uint32_t original[5][32];
        uint64_t results[32];
        for (auto &reg : original)
          for (uint32_t &value : reg)
            value = random();
        for (int lane = 0; lane < 32; ++lane) {
          uint64_t value = original[source[1]][lane] | (uint64_t(original[source[2]][lane]) << 32);
          results[lane] = goc_test::shift_reference(bits, op % 3, original[source[0]][lane], value);
        }
        for (uint64_t mask : rdna4_exec_masks())
          for (int low = 0; low < 5; ++low)
            for (int high = 0; high < (bits == 64 ? 5 : 1); ++high) {
              uint32_t words[5][32], expected[5][32];
              for (int reg = 0; reg < 5; ++reg) {
                std::copy_n(original[reg], 32, words[reg]);
                std::copy_n(original[reg], 32, expected[reg]);
              }
              const uint32_t *ap[] = {words[source[0]]},
                             *bp[] = {words[source[1]], words[source[2]]};
              uint32_t *dp[] = {words[low], words[high]};
              for (int lane = 0; lane < 32; ++lane)
                if ((mask >> lane) & 1) {
                  expected[low][lane] = uint32_t(results[lane]);
                  if (bits == 64)
                    expected[high][lane] = uint32_t(results[lane] >> 32);
                }
              ASSERT_EQ(functions[op](cpu, mask, 0, dp, ap, bp), GOC_SUCCESS);
              for (int reg = 0; reg < 5; ++reg)
                for (int lane = 0; lane < 32; ++lane)
                  ASSERT_EQ(words[reg][lane], expected[reg][lane])
                      << op << "/" << cpu << "/" << mask << "/" << low << "/" << high;
            }
      }
}

TEST(Shift, PreservesHostFloatingPointState) {
  std::fenv_t saved;
  ASSERT_EQ(std::fegetenv(&saved), 0);
#if defined(__x86_64__) || defined(_M_X64)
  unsigned saved_mxcsr = _mm_getcsr();
#endif
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO})
    for (int flush = 0; flush < 2; ++flush) {
      std::fesetround(rounding);
      std::feclearexcept(FE_ALL_EXCEPT);
      std::feraiseexcept(FE_DIVBYZERO);
#if defined(__x86_64__) || defined(_M_X64)
      _mm_setcsr((_mm_getcsr() & ~0x8040u) | (flush ? 0x8040u : 0));
      unsigned before = _mm_getcsr();
#endif
      for (Fn fn : functions)
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
          uint32_t a[32], b[2][32], d[2][32];
          const uint32_t *ap[] = {a}, *bp[] = {b[0], b[1]};
          uint32_t *dp[] = {d[0], d[1]};
          for (int lane = 0; lane < 32; ++lane) {
            a[lane] = uint32_t(lane * 7);
            b[0][lane] = 0x7f800001;
            b[1][lane] = 0xff800001;
          }
          EXPECT_EQ(fn(cpu, UINT32_MAX, 0, dp, ap, bp), GOC_SUCCESS);
        }
      EXPECT_EQ(std::fegetround(), rounding);
      EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
#if defined(__x86_64__) || defined(_M_X64)
      EXPECT_EQ(_mm_getcsr(), before);
#endif
    }
  std::fesetenv(&saved);
#if defined(__x86_64__) || defined(_M_X64)
  _mm_setcsr(saved_mxcsr);
#endif
}

TEST(Shift, RejectsModifiersAndStrictExactBeforeEmptyMask) {
  for (Fn fn : functions) {
    uint32_t a[32] = {}, b[2][32] = {}, d[2][32];
    const uint32_t *ap[] = {a}, *bp[] = {b[0], b[1]};
    uint32_t *dp[] = {d[0], d[1]};
    for (auto &reg : d)
      std::fill_n(reg, 32, 0xdeadbeef);
    for (uint64_t mask : {UINT64_C(0), UINT64_C(0xffffffff00000000), UINT64_MAX}) {
      for (int bit = 0; bit < 32; ++bit)
        EXPECT_EQ(fn(0, mask, uint32_t(1) << bit, dp, ap, bp), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(UINT64_C(1) << 63, mask, 0, dp, ap, bp), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, mask, 0, dp, ap, bp),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
    }
    for (const auto &reg : d)
      for (uint32_t value : reg)
        EXPECT_EQ(value, 0xdeadbeef);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_FP16_OVFL, UINT32_MAX, 0, dp, ap, bp),
              GOC_SUCCESS);
  }
}

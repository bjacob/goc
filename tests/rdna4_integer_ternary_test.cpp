// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_exec_masks.h"
#include "rdna4_integer_ternary_reference.h"

#include <algorithm>
#include <cfenv>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace {

using Fn = decltype(&goc_rdna4_v_lshl_add_u32);
const Fn functions[] = {goc_rdna4_v_lshl_add_u32, goc_rdna4_v_add_lshl_u32, goc_rdna4_v_lshl_or_b32,
                        goc_rdna4_v_and_or_b32,   goc_rdna4_v_or3_b32,      goc_rdna4_v_xor3_b32,
                        goc_rdna4_v_xad_u32,      goc_rdna4_v_lerp_u8};

::testing::AssertionResult check(int op, uint64_t flags, uint32_t words[4][32]) {
  uint32_t expected[32];
  for (int lane = 0; lane < 32; ++lane)
    expected[lane] =
        goc_test::integer_ternary_reference(op, words[0][lane], words[1][lane], words[2][lane]);
  const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
  uint32_t *d[] = {words[3]};
  int status = functions[op](flags, UINT32_MAX, 0, d, a, b, c);
  if (status != GOC_SUCCESS)
    return ::testing::AssertionFailure() << "status " << status;
  for (int lane = 0; lane < 32; ++lane)
    if (words[3][lane] != expected[lane])
      return ::testing::AssertionFailure() << op << "/" << flags << "/" << lane << ": "
                                           << words[3][lane] << " != " << expected[lane];
  return ::testing::AssertionSuccess();
}

} // namespace

TEST(IntegerTernary, BoundaryTriplesAndRandomValues) {
  const uint32_t edges[] = {0,          1,          31,         32,         63,         64,
                            0x7fffffff, 0x80000000, 0xffffffff, 0xfffffffe, 0xff00ff00, 0x00ff00ff,
                            0x01010101, 0xfefefefe, 0x55555555, 0xaaaaaaaa};
  for (int op = 0; op < 8; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      std::mt19937 random(1023);
      for (int start = 0; start < 8192; start += 32) {
        uint32_t words[4][32];
        for (int lane = 0; lane < 32; ++lane) {
          int index = start + lane;
          for (int reg = 0; reg < 3; ++reg) {
            words[reg][lane] = start < 4096 ? edges[index % 16] : random();
            index /= 16;
          }
        }
        ASSERT_TRUE(check(op, cpu, words));
      }
    }
}

TEST(IntegerTernary, EveryShiftCountAndBitPosition) {
  for (int op = 0; op < 3; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int bit = 0; bit < 32; ++bit)
        for (bool inverted : {false, true})
          for (uint32_t start = 0; start < 256; start += 32) {
            uint32_t words[4][32];
            for (uint32_t lane = 0; lane < 32; ++lane) {
              words[0][lane] = (uint32_t(1) << bit) ^ (inverted ? UINT32_MAX : 0);
              uint32_t count = (start + lane) | (lane % 2 ? 0x80000000 : 0xffff0000);
              words[1][lane] = op == 1 ? 0xf0000001 : count;
              words[2][lane] = op == 1 ? count : 0x33333333;
            }
            ASSERT_TRUE(check(op, cpu, words));
          }
}

TEST(IntegerTernary, BooleanTruthTablesAtEveryBit) {
  for (int op = 3; op < 7; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int truth = 0; truth < 8; ++truth)
        for (bool surrounding : {false, true}) {
          uint32_t words[4][32];
          for (int lane = 0; lane < 32; ++lane)
            for (int reg = 0; reg < 3; ++reg) {
              uint32_t bit = uint32_t(1) << lane;
              words[reg][lane] = (surrounding ? ~bit : 0) | ((truth >> reg) & 1 ? bit : 0);
            }
          ASSERT_TRUE(check(op, cpu, words));
        }
}

TEST(IntegerTernary, EveryBytePairAndRoundingControl) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (int rounding = 0; rounding < 16; ++rounding)
      for (unsigned start = 0; start < 65536; start += 32) {
        uint32_t words[4][32];
        for (unsigned lane = 0; lane < 32; ++lane) {
          unsigned a = (start + lane) / 256, b = (start + lane) % 256;
          words[0][lane] = a | ((255 - a) << 8) | ((a ^ 128) << 16) | ((a ^ 1) << 24);
          words[1][lane] = b | ((255 - b) << 8) | ((b ^ 128) << 16) | ((b ^ 1) << 24);
          words[2][lane] = ((start + lane) * 0x9e3779b9u) & 0xfefefefe;
          for (int byte = 0; byte < 4; ++byte)
            words[2][lane] |= uint32_t((rounding >> byte) & 1) << (8 * byte);
        }
        ASSERT_TRUE(check(7, cpu, words));
      }
}

TEST(IntegerTernary, LiteralStageOrderAndByteOverflow) {
  const uint32_t cases[][4] = {{0xffffffff, 1, 3, 1},
                               {0xffffffff, 1, 31, 0},
                               {3, 1, 2, 6},
                               {0x0f0f0f0f, 0x33333333, 0x80808080, 0x83838383},
                               {0x0f0f0f0f, 0x33333333, 0x80808080, 0xbfbfbfbf},
                               {0x0f0f0f0f, 0x33333333, 0x80808080, 0xbcbcbcbc},
                               {0xffffffff, 1, 3, 1},
                               {0xff00ff00, 0x00ff00ff, 0x01000100, 0x807f807f}};
  for (int op = 0; op < 8; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t words[4][32];
      for (int reg = 0; reg < 3; ++reg)
        std::fill_n(words[reg], 32, cases[op][reg]);
      const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
      uint32_t *d[] = {words[3]};
      ASSERT_EQ(functions[op](cpu, UINT32_MAX, 0, d, a, b, c), GOC_SUCCESS);
      for (uint32_t value : words[3])
        EXPECT_EQ(value, cases[op][3]);
    }
}

TEST(IntegerTernary, MasksAndWholeRegisterAliases) {
  const int sources[][3] = {{0, 1, 2}, {0, 0, 2}, {0, 1, 0}, {0, 1, 1}, {0, 0, 0}};
  for (int op = 0; op < 8; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (const auto &source : sources) {
        std::mt19937 random(560);
        uint32_t original[4][32], result[32];
        for (auto &reg : original)
          for (auto &word : reg)
            word = random();
        for (int lane = 0; lane < 32; ++lane)
          result[lane] = goc_test::integer_ternary_reference(
              op, original[source[0]][lane], original[source[1]][lane], original[source[2]][lane]);
        for (uint64_t mask : rdna4_exec_masks())
          for (int target = 0; target < 4; ++target) {
            uint32_t words[4][32], expected[4][32];
            for (int reg = 0; reg < 4; ++reg) {
              std::copy_n(original[reg], 32, words[reg]);
              std::copy_n(original[reg], 32, expected[reg]);
            }
            for (int lane = 0; lane < 32; ++lane)
              if ((mask >> lane) & 1)
                expected[target][lane] = result[lane];
            const uint32_t *a[] = {words[source[0]]}, *b[] = {words[source[1]]},
                           *c[] = {words[source[2]]};
            uint32_t *d[] = {words[target]};
            ASSERT_EQ(functions[op](cpu, mask, 0, d, a, b, c), GOC_SUCCESS);
            for (int reg = 0; reg < 4; ++reg)
              ASSERT_TRUE(std::equal(words[reg], words[reg] + 32, expected[reg]));
          }
      }
}

TEST(IntegerTernary, ValidationAndHostFpState) {
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
      for (int op = 0; op < 8; ++op)
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
          uint32_t words[4][32];
          for (auto &reg : words)
            std::fill_n(reg, 32, 0x7f800001);
          const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
          uint32_t *d[] = {words[3]};
          for (uint64_t mask : {UINT64_C(0), UINT64_MAX}) {
            for (int bit = 0; bit < 32; ++bit)
              EXPECT_EQ(functions[op](cpu, mask, uint32_t(1) << bit, d, a, b, c),
                        GOC_ERROR_INVALID_FLAGS);
            EXPECT_EQ(functions[op](cpu | (UINT64_C(1) << 63), mask, 0, d, a, b, c),
                      GOC_ERROR_INVALID_FLAGS);
            EXPECT_EQ(functions[op](cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                                    mask, 0, d, a, b, c),
                      GOC_ERROR_UNSUPPORTED_SEMANTICS);
          }
          for (const auto &reg : words)
            for (uint32_t value : reg)
              EXPECT_EQ(value, 0x7f800001);
          EXPECT_TRUE(check(op, cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_FP16_OVFL, words));
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

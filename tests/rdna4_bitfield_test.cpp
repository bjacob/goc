// SPDX-License-Identifier: MIT

#include "fp_environment.h"
#include "goc/goc.h"
#include "rdna4_bitfield_reference.h"
#include "rdna4_exec_masks.h"

#include <algorithm>
#include <cfenv>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace {

::testing::AssertionResult check(int op, uint64_t flags, uint32_t words[4][32]) {
  uint32_t expected[32];
  for (int lane = 0; lane < 32; ++lane)
    expected[lane] =
        goc_test::bitfield_reference(op, words[0][lane], words[1][lane], words[2][lane]);
  const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
  uint32_t *d[] = {words[3]};
  int status = goc_test::bitfield_functions[op](flags, UINT32_MAX, 0, d, a, b, c);
  if (status != GOC_SUCCESS)
    return ::testing::AssertionFailure() << "status " << status;
  for (int lane = 0; lane < 32; ++lane)
    if (words[3][lane] != expected[lane])
      return ::testing::AssertionFailure() << op << "/" << flags << "/" << lane << ": "
                                           << words[3][lane] << " != " << expected[lane];
  return ::testing::AssertionSuccess();
}

} // namespace

TEST(Bitfield, BoundaryTriplesAndRandomValues) {
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

TEST(Bitfield, EveryOffsetWidthAndSourceBit) {
  for (int op : {0, 1, 3})
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned bit = 0; bit < 32; ++bit)
        for (bool inverted : {false, true})
          for (unsigned width = 0; width < 64; ++width) {
            uint32_t words[4][32];
            for (unsigned lane = 0; lane < 32; ++lane) {
              words[0][lane] =
                  op == 3 ? width | 0x80000000 : (uint32_t(1) << bit) ^ (inverted ? UINT32_MAX : 0);
              words[1][lane] = lane | 0xffffffe0;
              words[2][lane] = width | 0x80000000;
            }
            ASSERT_TRUE(check(op, cpu, words));
          }
}

TEST(Bitfield, InsertionTruthTableAndReversal) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (int truth = 0; truth < 8; ++truth)
      for (bool inverted : {false, true}) {
        uint32_t words[4][32];
        for (int lane = 0; lane < 32; ++lane)
          for (int reg = 0; reg < 3; ++reg) {
            uint32_t bit = uint32_t(1) << lane;
            words[reg][lane] = (inverted ? ~bit : 0) | ((truth >> reg) & 1 ? bit : 0);
          }
        ASSERT_TRUE(check(2, cpu, words));
        ASSERT_TRUE(check(4, cpu, words));
      }
}

TEST(Bitfield, LiteralBoundaryResults) {
  // Operation, A, B, C, expected result.
  const uint32_t cases[][5] = {{0, 0x80000000, 31, 31, 1},
                               {1, 0x80000000, 31, 31, 0xffffffff},
                               {0, 0xffffffff, 0, 32, 0},
                               {1, 0xffffffff, 0, 32, 0},
                               {1, 0x40000000, 30, 1, 0xffffffff},
                               {1, 0x40000000, 30, 2, 1},
                               {2, 0x55555555, 0xffffffff, 0, 0x55555555},
                               {3, 31, 31, 0, 0x80000000},
                               {3, 32, 0, 0, 0},
                               {4, 0x01234567, 0, 0, 0xe6a2c480}};
  for (const auto &item : cases)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t words[4][32];
      for (int reg = 0; reg < 3; ++reg)
        std::fill_n(words[reg], 32, item[reg + 1]);
      const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
      uint32_t *d[] = {words[3]};
      ASSERT_EQ(goc_test::bitfield_functions[item[0]](cpu, UINT32_MAX, 0, d, a, b, c), GOC_SUCCESS);
      for (uint32_t value : words[3])
        EXPECT_EQ(value, item[4]);
    }
}

TEST(Bitfield, MasksAndWholeRegisterAliases) {
  const int sources[][3] = {{0, 1, 2}, {0, 0, 2}, {0, 1, 0}, {0, 1, 1}, {0, 0, 0}};
  for (int op = 0; op < 8; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (const auto &source : sources) {
        std::mt19937 random(560);
        uint32_t original[4][32], result[32];
        for (auto &reg : original)
          for (auto &word : reg)
            word = random();
        if (op == 7)
          for (auto &word : original[source[2]])
            word &= 0x0f0f0f0f;
        for (int lane = 0; lane < 32; ++lane)
          result[lane] = goc_test::bitfield_reference(
              op, original[source[0]][lane], original[source[1]][lane], original[source[2]][lane]);
        for (uint32_t mask : rdna4_exec_masks())
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
            ASSERT_EQ(goc_test::bitfield_functions[op](cpu, mask, 0, d, a, b, c), GOC_SUCCESS);
            for (int reg = 0; reg < 4; ++reg)
              ASSERT_TRUE(std::equal(words[reg], words[reg] + 32, expected[reg]));
          }
      }
}

TEST(Bitfield, ValidationAndHostFpState) {
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
          for (uint32_t mask : {UINT32_C(0), UINT32_MAX}) {
            for (int bit = 0; bit < 32; ++bit)
              EXPECT_EQ(goc_test::bitfield_functions[op](cpu, mask, uint32_t(1) << bit, d, a, b, c),
                        GOC_ERROR_INVALID_FLAGS);
            EXPECT_EQ(
                goc_test::bitfield_functions[op](cpu | (UINT64_C(1) << 63), mask, 0, d, a, b, c),
                GOC_ERROR_INVALID_FLAGS);
            EXPECT_EQ(goc_test::bitfield_functions[op](cpu | GOC_SEMANTICS_EXACT_EMPIRICAL |
                                                           GOC_SEMANTICS_STRICT,
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

TEST(Bitfield, AlignmentEveryConcatenatedBitAndShift) {
  for (int op : {5, 6})
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned bit = 0; bit < 64; ++bit)
        for (bool inverted : {false, true}) {
          uint64_t source = (UINT64_C(1) << bit) ^ (inverted ? UINT64_MAX : 0);
          uint32_t words[4][32];
          for (unsigned lane = 0; lane < 32; ++lane) {
            words[0][lane] = uint32_t(source >> 32);
            words[1][lane] = uint32_t(source);
            // Ignored high bits must not turn a wrapped shift into a zero result.
            words[2][lane] = 0xffffffe0u | lane;
          }
          ASSERT_TRUE(check(op, cpu, words));
        }
}

TEST(Bitfield, PermuteEverySelectorAndSourceBit) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned bit = 0; bit < 64; ++bit)
      for (bool invert : {false, true})
        for (unsigned start = 0; start < 256; start += 32) {
          uint32_t words[4][32];
          for (unsigned lane = 0; lane < 32; ++lane) {
            uint64_t source = (UINT64_C(1) << ((bit + lane) % 64)) ^ (invert ? UINT64_MAX : 0);
            words[0][lane] = uint32_t(source >> 32);
            words[1][lane] = uint32_t(source);
            words[2][lane] = (start + lane) * 0x01010101u;
          }
          ASSERT_TRUE(check(7, cpu, words));
        }
}

TEST(Bitfield, PermuteHardwareAllSelectorBytes) {
  // GFX1201 capture: 65536 pseudorandom source pairs. Each byte position
  // receives every selector 256 times. Digest is FNV-1a, low byte first.
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint64_t digest = UINT64_C(14695981039346656037);
    for (unsigned start = 0; start < 65536; start += 32) {
      uint32_t words[4][32];
      for (unsigned lane = 0; lane < 32; ++lane) {
        unsigned i = start + lane;
        words[0][lane] = (i * 0x7395a831u) ^ 0xa7925163u;
        words[1][lane] = (i * 0x83a1459du) ^ 0x5389d241u;
        words[2][lane] = 0;
        for (unsigned byte = 0; byte < 4; ++byte)
          words[2][lane] |= ((i + 85 * byte) & 255) << (8 * byte);
      }
      const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
      uint32_t *d[] = {words[3]};
      ASSERT_EQ(goc_rdna4_v_perm_b32(cpu, UINT32_MAX, 0, d, a, b, c), GOC_SUCCESS);
      for (uint32_t value : words[3])
        for (unsigned byte = 0; byte < 4; ++byte) {
          digest ^= (value >> (8 * byte)) & 255;
          digest *= UINT64_C(1099511628211);
        }
    }
    EXPECT_EQ(digest, UINT64_C(0x130dcc447fe090a6)) << cpu;
  }
}

TEST(Bitfield, UnalignedStorageAndEmptyExec) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned op = 0; op < 8; ++op) {
      EXPECT_EQ(
          goc_test::bitfield_functions[op](cpu, UINT32_C(0), 0, nullptr, nullptr, nullptr, nullptr),
          GOC_SUCCESS);
      for (unsigned target = 0; target < 4; ++target)
        for (uint32_t mask : rdna4_exec_masks()) {
          uint32_t words[4][35], expected[4][35];
          for (unsigned reg = 0; reg < 4; ++reg)
            for (unsigned lane = 0; lane < 35; ++lane) {
              uint32_t value = (lane * 0x7395a831u) ^ (reg * 0xd317b579u);
              if (reg == 2)
                value &= 0x0f0f0f0f;
              words[reg][lane] = expected[reg][lane] = value;
            }
          for (unsigned lane = 0; lane < 32; ++lane)
            if ((mask >> lane) & 1)
              expected[target][lane + 1] = goc_test::bitfield_reference(
                  op, words[0][lane + 1], words[1][lane + 1], words[2][lane + 1]);
          const uint32_t *a[] = {words[0] + 1}, *b[] = {words[1] + 1}, *c[] = {words[2] + 1};
          uint32_t *d[] = {words[target] + 1};
          ASSERT_EQ(goc_test::bitfield_functions[op](cpu, mask, 0, d, a, b, c), GOC_SUCCESS);
          for (unsigned reg = 0; reg < 4; ++reg)
            for (unsigned lane = 0; lane < 35; ++lane)
              ASSERT_EQ(words[reg][lane], expected[reg][lane]) << op << "/" << cpu << "/" << lane;
        }
    }
}

TEST(Bitfield, EveryShiftPreservesFpState) {
  goc_test::ScopedFpEnvironment restore;
  ASSERT_TRUE(restore.saved());

  for (unsigned op = 0; op < 8; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        SCOPED_TRACE(::testing::Message() << op << '/' << cpu << '/' << rounding);
        uint32_t a[32], b[32], c[32], d[32];
        for (unsigned lane = 0; lane < 32; ++lane) {
          a[lane] = 0xffffffe0u | lane;
          b[lane] = lane;
          c[lane] = 31 - lane;
        }
        auto pa = a, pb = b, pc = c, pd = d;
        ASSERT_EQ(std::fesetround(rounding), 0);
        ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
        ASSERT_EQ(std::feraiseexcept(FE_DIVBYZERO), 0);
        ASSERT_EQ(goc_test::bitfield_functions[op](cpu, UINT32_MAX, 0, &pd, &pa, &pb, &pc),
                  GOC_SUCCESS);
        EXPECT_EQ(std::fegetround(), rounding);
        EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
      }
}

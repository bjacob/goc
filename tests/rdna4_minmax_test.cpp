// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "internal.h"
#include "rdna4_exec_masks.h"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_rdna4_v_fma_f32);
const Fn functions[] = {
    goc_rdna4_v_min3_num_f32,       goc_rdna4_v_max3_num_f32,       goc_rdna4_v_minmax_num_f32,
    goc_rdna4_v_maxmin_num_f32,     goc_rdna4_v_minimum3_f32,       goc_rdna4_v_maximum3_f32,
    goc_rdna4_v_minimummaximum_f32, goc_rdna4_v_maximumminimum_f32, goc_rdna4_v_med3_num_f32};

bool nan(uint32_t x) { return (x & 0x7fffffff) > 0x7f800000; }

// Sortable integer keys distinguish signed zeros without host min/max rules.
uint32_t ordered(uint32_t x) { return x & 0x80000000 ? ~x : x ^ 0x80000000; }

uint32_t select(uint32_t a, uint32_t b, bool maximum, bool propagate) {
  if (propagate) {
    for (uint32_t x : {a, b})
      if (nan(x) && !(x & 0x00400000))
        return x | 0x00400000;
    for (uint32_t x : {a, b})
      if (nan(x))
        return x;
  } else {
    if (nan(a) && nan(b))
      return a | 0x00400000;
    if (nan(a) || nan(b))
      return nan(a) ? b : a;
  }
  return (maximum ? ordered(a) > ordered(b) : ordered(a) < ordered(b)) ? a : b;
}

uint32_t reference(int op, uint32_t a, uint32_t b, uint32_t c, uint32_t mode) {
  uint32_t inputs[] = {a, b, c};
  for (int i = 0; i < 3; ++i) {
    if (mode & (GOC_ALU_ABS_A << i))
      inputs[i] &= 0x7fffffff;
    if (mode & (GOC_ALU_NEG_A << i))
      inputs[i] ^= 0x80000000;
  }
  bool first_max = op % 4 == 1 || op % 4 == 3;
  bool second_max = op % 4 == 1 || op % 4 == 2;
  uint32_t result =
      select(select(inputs[0], inputs[1], first_max, op >= 4), inputs[2], second_max, op >= 4);
  if (op == 8) {
    if (nan(inputs[0]) || nan(inputs[1]) || nan(inputs[2])) {
      result = select(select(inputs[0], inputs[1], false, false), inputs[2], false, false);
    } else {
      // Sorting independently establishes the median for nonzero values. The
      // ISA's first-maximum removal rule additionally defines signed-zero ties.
      uint32_t sorted[] = {inputs[0], inputs[1], inputs[2]};
      std::sort(sorted, sorted + 3, [](uint32_t a, uint32_t b) { return ordered(a) < ordered(b); });
      result = sorted[1];
      if ((result & 0x7fffffff) == 0) {
        float maximum = goc::as_float(sorted[2]);
        int drop = goc::as_float(inputs[0]) == maximum   ? 0
                   : goc::as_float(inputs[1]) == maximum ? 1
                                                         : 2;
        result = select(inputs[(drop + 1) % 3], inputs[(drop + 2) % 3], true, false);
      }
    }
  }
  const int exponents[] = {0, 1, 2, -1};
  float value = std::ldexp(goc::as_float(result), exponents[(mode >> 6) & 3]);
  if (mode & GOC_ALU_CLAMP)
    value = !(value > 0) ? 0 : std::min(value, 1.0f);
  return goc::as_bits(value);
}

} // namespace

TEST(Minmax3, AllModifiersMasksAliasesAndSpecialValues) {
  const uint32_t values[] = {0,          0x80000000, 1,          0x80000001, 0x007fffff, 0x00800000,
                             0x3f000000, 0xbf000000, 0x3f800000, 0xbf800000, 0x3f800001, 0x3f7fffff,
                             0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc12345, 0xffc12345,
                             0x3e800000, 0x40000000, 0xc0400000, 0x7f812345, 0xff812346};
  for (int op = 0; op < 9; ++op)
    for (uint32_t mode = 0; mode < 512; ++mode) {
      uint32_t source[3][32], expected[32];
      for (int lane = 0; lane < 32; ++lane) {
        for (int reg = 0; reg < 3; ++reg)
          source[reg][lane] = values[(lane * (2 * reg + 1) + reg * 3) % 23];
        expected[lane] = reference(op, source[0][lane], source[1][lane], source[2][lane], mode);
      }
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (uint64_t mask : rdna4_exec_masks())
          for (int alias = 0; alias < 4; ++alias) {
            SCOPED_TRACE(::testing::Message()
                         << op << "/" << mode << "/" << cpu << "/" << mask << "/" << alias);
            uint32_t storage[4][34], before[32];
            uint32_t *v[4];
            for (int reg = 0; reg < 4; ++reg) {
              std::fill(storage[reg], storage[reg] + 34, 0xdeadbeef);
              v[reg] = storage[reg] + 1;
              if (reg < 3)
                std::copy(source[reg], source[reg] + 32, v[reg]);
            }
            std::copy(v[alias], v[alias] + 32, before);
            ASSERT_EQ(functions[op](cpu, mask, mode, &v[alias], &v[0], &v[1], &v[2]), GOC_SUCCESS);
            for (int lane = 0; lane < 32; ++lane) {
              uint32_t want = (mask >> lane) & 1 ? expected[lane] : before[lane];
              EXPECT_EQ(v[alias][lane], want);
            }
            for (const auto &reg : storage) {
              EXPECT_EQ(reg[0], 0xdeadbeef);
              EXPECT_EQ(reg[33], 0xdeadbeef);
            }
          }
    }
}

TEST(Minmax3, LiteralOrderNaNsAndSignedZeros) {
  // Inputs A/B/C followed by all eight results in public-function order.
  const uint32_t cases[][11] = {
      {0x3f800000, 0x40000000, 0x40400000, 0x3f800000, 0x40400000, 0x40400000, 0x40000000,
       0x3f800000, 0x40400000, 0x40400000, 0x40000000},
      {0x40400000, 0x40000000, 0x3f800000, 0x3f800000, 0x40400000, 0x40000000, 0x3f800000,
       0x3f800000, 0x40400000, 0x40000000, 0x3f800000},
      {0x80000000, 0, 0x80000000, 0x80000000, 0, 0x80000000, 0x80000000, 0x80000000, 0, 0x80000000,
       0x80000000},
      {0, 0x80000000, 0, 0x80000000, 0, 0, 0, 0x80000000, 0, 0, 0},
      {0x7f800001, 0xff800002, 0x7f800003, 0x7fc00001, 0x7fc00001, 0x7fc00001, 0x7fc00001,
       0x7fc00003, 0x7fc00003, 0x7fc00003, 0x7fc00003},
      {0x7fc00001, 0xff800002, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000,
       0xffc00002, 0xffc00002, 0xffc00002, 0xffc00002},
      {0x7fc00001, 0x40000000, 0x40400000, 0x40000000, 0x40400000, 0x40400000, 0x40000000,
       0x7fc00001, 0x7fc00001, 0x7fc00001, 0x7fc00001},
  };
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (int op = 0; op < 8; ++op)
      for (const auto &test : cases) {
        SCOPED_TRACE(::testing::Message()
                     << cpu << "/" << op << "/" << test[0] << "/" << test[1] << "/" << test[2]);
        uint32_t a[32], b[32], c[32], d[32];
        std::fill(a, a + 32, test[0]);
        std::fill(b, b + 32, test[1]);
        std::fill(c, c + 32, test[2]);
        auto pa = a, pb = b, pc = c, pd = d;
        ASSERT_EQ(functions[op](cpu, UINT32_MAX, 0, &pd, &pa, &pb, &pc), GOC_SUCCESS);
        for (uint32_t value : d)
          EXPECT_EQ(value, test[op + 3]);
      }
}

TEST(Minmax3, RandomBitPatterns) {
  std::mt19937 random(20261008);
  for (int batch = 0; batch < 128; ++batch) {
    uint32_t a[32], b[32], c[32], d[32];
    for (int lane = 0; lane < 32; ++lane) {
      a[lane] = random();
      b[lane] = random();
      c[lane] = random();
    }
    auto pa = a, pb = b, pc = c, pd = d;
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int op = 0; op < 9; ++op) {
        SCOPED_TRACE(::testing::Message() << batch << "/" << cpu << "/" << op);
        ASSERT_EQ(functions[op](cpu, UINT32_MAX, 0, &pd, &pa, &pb, &pc), GOC_SUCCESS);
        for (int lane = 0; lane < 32; ++lane)
          EXPECT_EQ(d[lane], reference(op, a[lane], b[lane], c[lane], 0));
      }
  }
}

TEST(Minmax3, Validation) {
  for (auto fn : functions) {
    uint32_t a[32] = {}, d[32];
    std::fill(d, d + 32, 0xdeadbeef);
    auto pa = a, pd = d;
    EXPECT_EQ(fn(0, UINT32_MAX, GOC_ALU_HIGH_C, &pd, &pa, &pa, &pa), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(fn(0, 0, GOC_ALU_HIGH_D, &pd, &pa, &pa, &pa), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0, &pd, &pa, &pa, &pa),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL, 0xffffffff00000000, 0x1ff, &pd, &pa, &pa, &pa),
              GOC_SUCCESS);
    for (uint32_t value : d)
      EXPECT_EQ(value, 0xdeadbeef);
  }
}

TEST(Minmax3, MedianLiteralNaNsAndZeroTies) {
  const uint32_t cases[][4] = {
      {0x40400000, 0x3f800000, 0x40000000, 0x40000000},
      {0x7f800000, 0xff800000, 0x3f800000, 0x3f800000},
      {0x7fc12345, 0x40000000, 0x3f800000, 0x3f800000},
      {0x40000000, 0x7f812345, 0x3f800000, 0x3f800000},
      {0x40000000, 0x3f800000, 0xff812345, 0x3f800000},
      {0xff812345, 0x7fc12346, 0x7f812347, 0xffc12345},
      {0, 0x80000000, 0x80000000, 0x80000000},
      {0x80000000, 0, 0x80000000, 0},
      {0x80000000, 0x80000000, 0, 0},
      {0x80000000, 0x80000000, 0x80000000, 0x80000000},
      {0, 0, 0x80000000, 0},
      {0x3f800000, 0, 0x80000000, 0},
  };
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (const auto &test : cases) {
      uint32_t a[32], b[32], c[32], d[32];
      std::fill(a, a + 32, test[0]);
      std::fill(b, b + 32, test[1]);
      std::fill(c, c + 32, test[2]);
      auto pa = a, pb = b, pc = c, pd = d;
      ASSERT_EQ(goc_rdna4_v_med3_num_f32(cpu, UINT32_MAX, 0, &pd, &pa, &pb, &pc), GOC_SUCCESS);
      for (uint32_t value : d)
        EXPECT_EQ(value, test[3]) << cpu << "/" << test[0] << "/" << test[1] << "/" << test[2];
    }
}

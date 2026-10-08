// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "internal.h"
#include "rdna4_exec_masks.h"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_rdna4_v_add_f32);
const Fn functions[] = {goc_rdna4_v_add_f32,     goc_rdna4_v_sub_f32,     goc_rdna4_v_subrev_f32,
                        goc_rdna4_v_mul_f32,     goc_rdna4_v_min_num_f32, goc_rdna4_v_max_num_f32,
                        goc_rdna4_v_minimum_f32, goc_rdna4_v_maximum_f32};

float reference(int op, float a, float b, uint32_t mode) {
  double x = a, y = b;
  if (mode & GOC_ALU_ABS_A)
    x = std::abs(x);
  if (mode & GOC_ALU_ABS_B)
    y = std::abs(y);
  if (mode & GOC_ALU_NEG_A)
    x = -x;
  if (mode & GOC_ALU_NEG_B)
    y = -y;
  float value = float(op == 0 ? x + y : op == 1 ? x - y : op == 2 ? y - x : x * y);
  if (op >= 4) {
    bool maximum = op == 5 || op == 7;
    if (std::isnan(x) || std::isnan(y))
      value = op >= 6 ? NAN : std::isnan(x) ? y : x;
    else if (x == 0 && y == 0)
      value =
          (maximum ? (std::signbit(x) && std::signbit(y)) : (std::signbit(x) || std::signbit(y)))
              ? -0.0f
              : 0.0f;
    else
      value = maximum ? std::max(x, y) : std::min(x, y);
  }
  const float scales[] = {1, 2, 4, 0.5f};
  value *= scales[(mode >> 6) & 3];
  if (mode & GOC_ALU_CLAMP)
    value = !(value > 0) ? 0 : std::min(value, 1.0f);
  return value;
}

} // namespace

TEST(Binary, AllModifiersMasksAliasesAndSpecialValues) {
  const uint32_t values[] = {0,          0x80000000, 1,          0x80000001, 0x007fffff, 0x00800000,
                             0x3f000000, 0xbf000000, 0x3f800000, 0xbf800000, 0x3f800001, 0x3f7fffff,
                             0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc12345, 0xffc12345,
                             0x3e800000, 0x40000000, 0xc0400000};
  for (int op = 0; op < 8; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint32_t variant = 0; variant < 128; ++variant) {
        uint32_t mode = (variant & 3) | ((variant & 12) << 1) | ((variant & 112) << 2);
        for (uint64_t mask : rdna4_exec_masks())
          for (int alias = 0; alias < 3; ++alias) {
            SCOPED_TRACE(::testing::Message()
                         << op << "/" << cpu << "/" << mode << "/" << mask << "/" << alias);
            uint32_t storage[3][34], before[32];
            uint32_t *v[3];
            for (int reg = 0; reg < 3; ++reg) {
              std::fill(storage[reg], storage[reg] + 34, 0xdeadbeef);
              v[reg] = storage[reg] + 1;
            }
            for (int lane = 0; lane < 32; ++lane) {
              v[0][lane] = values[lane % 21];
              v[1][lane] = values[(lane * 5 + 3) % 21];
              before[lane] = v[alias][lane];
            }
            ASSERT_EQ(functions[op](cpu, mask, mode, &v[alias], &v[0], &v[1]), GOC_SUCCESS);
            for (int lane = 0; lane < 32; ++lane) {
              if (!((mask >> lane) & 1)) {
                EXPECT_EQ(v[alias][lane], before[lane]);
                continue;
              }
              float want = reference(op, goc::as_float(values[lane % 21]),
                                     goc::as_float(values[(lane * 5 + 3) % 21]), mode);
              if (std::isnan(want)) {
                EXPECT_TRUE(std::isnan(goc::as_float(v[alias][lane])));
              } else {
                EXPECT_EQ(v[alias][lane], goc::as_bits(want));
              }
            }
            for (const auto &reg : storage) {
              EXPECT_EQ(reg[0], 0xdeadbeef);
              EXPECT_EQ(reg[33], 0xdeadbeef);
            }
          }
      }
}

TEST(Binary, Validation) {
  for (auto fn : functions) {
    uint32_t a[32] = {}, d[32];
    std::fill(d, d + 32, 0xdeadbeef);
    auto pa = a, pd = d;
    EXPECT_EQ(fn(0, UINT32_MAX, GOC_ALU_NEG_C, &pd, &pa, &pa), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(fn(0, 0, GOC_ALU_HIGH_C, &pd, &pa, &pa), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0, &pd, &pa, &pa),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL, 0xffffffff00000000, 0, &pd, &pa, &pa), GOC_SUCCESS);
    for (auto value : d)
      EXPECT_EQ(value, 0xdeadbeef);
  }
}

TEST(Binary, MinMaxLiteralNaNsAndSignedZeros) {
  // Expected columns: minimumNumber, maximumNumber, minimum, maximum.
  const uint32_t cases[][6] = {
      {0, 0x80000000, 0x80000000, 0, 0x80000000, 0},
      {0x80000000, 0, 0x80000000, 0, 0x80000000, 0},
      {0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000},
      {0x7fc12345, 0xbf800000, 0xbf800000, 0xbf800000, 0x7fc12345, 0x7fc12345},
      {0xbf800000, 0xff812345, 0xbf800000, 0xbf800000, 0xffc12345, 0xffc12345},
      {0x7f812345, 0x3f800000, 0x3f800000, 0x3f800000, 0x7fc12345, 0x7fc12345},
      {0x3f800000, 0xffc12345, 0x3f800000, 0x3f800000, 0xffc12345, 0xffc12345},
      {0x7fc12345, 0xff812346, 0x7fc12345, 0x7fc12345, 0xffc12346, 0xffc12346},
      {0x7f812345, 0xff812346, 0x7fc12345, 0x7fc12345, 0x7fc12345, 0x7fc12345},
      {0xffc12345, 0x7fc12346, 0xffc12345, 0xffc12345, 0xffc12345, 0xffc12345},
      {0xff800000, 0x7f800000, 0xff800000, 0x7f800000, 0xff800000, 0x7f800000},
      {1, 0x80000001, 0x80000001, 1, 0x80000001, 1},
  };
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (int op = 0; op < 4; ++op)
      for (const auto &test : cases) {
        SCOPED_TRACE(::testing::Message() << cpu << "/" << op << "/" << test[0] << "/" << test[1]);
        uint32_t a[32], b[32], d[32];
        std::fill(a, a + 32, test[0]);
        std::fill(b, b + 32, test[1]);
        auto pa = a, pb = b, pd = d;
        ASSERT_EQ(functions[op + 4](cpu, UINT32_MAX, 0, &pd, &pa, &pb), GOC_SUCCESS);
        for (uint32_t value : d)
          EXPECT_EQ(value, test[op + 2]);
      }
}

// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_exec_masks.h"
#include "rdna4_omod_reference.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_rdna4_v_sqrt_f64);
const Fn functions[] = {goc_rdna4_v_trunc_f64, goc_rdna4_v_ceil_f64,  goc_rdna4_v_rndne_f64,
                        goc_rdna4_v_floor_f64, goc_rdna4_v_fract_f64, goc_rdna4_v_sqrt_f64,
                        goc_rdna4_v_rcp_f64,   goc_rdna4_v_rsq_f64,   goc_rdna4_v_frexp_mant_f64};

uint64_t bits(double value) {
  uint64_t result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

double number(uint64_t value) {
  double result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

double reference(int op, uint64_t input, uint32_t mode) {
  long double x = number(input);
  if (mode & GOC_ALU_ABS_A)
    x = std::abs(x);
  if (mode & GOC_ALU_NEG_A)
    x = -x;
  long double y = 0;
  switch (op) {
  case 8: {
    int exponent;
    y = std::isfinite(x) ? std::frexp(x, &exponent) : x;
    break;
  }
  case 0:
    y = std::trunc(x);
    break;
  case 1:
    y = std::ceil(x);
    break;
  case 2:
    y = x;
    if (std::isfinite(x) && x != 0) {
      long double lo = std::floor(x), fraction = x - lo;
      y = lo + (fraction > 0.5L || (fraction == 0.5L && std::fmod(lo, 2.0L) != 0));
      y = std::copysign(std::abs(y), x);
    }
    break;
  case 3:
    y = std::floor(x);
    break;
  case 4:
    y = x - std::floor(x);
    if (y > number(0x3fefffffffffffffULL))
      y = number(0x3fefffffffffffffULL);
    break;
  case 5:
    y = std::sqrt(x);
    break;
  case 6:
    y = 1 / x;
    break;
  case 7:
    y = 1 / std::sqrt(x);
    break;
  }
  double result = double(y);
  result = goc_test::omod_f64_reference(result, mode);
  if (mode & GOC_ALU_CLAMP)
    result = !(result > 0) ? 0 : std::min(result, 1.0);
  return result;
}

} // namespace

TEST(Fp64Unary, AllModifiersMasksAndCrossHalfAliases) {
  const uint64_t inputs[] = {0,
                             0x8000000000000000,
                             1,
                             0x8000000000000001,
                             0x0010000000000000,
                             0x8010000000000000,
                             0x3fe0000000000000,
                             0xbfe0000000000000,
                             0x3ff8000000000000,
                             0xbff8000000000000,
                             0x4004000000000000,
                             0xc004000000000000,
                             0x400c000000000000,
                             0xc00c000000000000,
                             0x3fefffffffffffff,
                             0x3ff0000000000001,
                             0x7fefffffffffffff,
                             0xffefffffffffffff,
                             0x7ff0000000000000,
                             0xfff0000000000000,
                             0x7ff8000000001234,
                             0xfff8000000001234,
                             0x7ff0000000001234,
                             0xfff0000000001234,
                             0x4330000000000001,
                             0xc330000000000001,
                             0x3fe0000000000001,
                             0x3fdfffffffffffff,
                             0x4010000000000000,
                             0x3fd0000000000000,
                             0x000fffffffffffff,
                             0x800fffffffffffff};
  const int aliases[][2] = {{2, 3}, {0, 1}, {1, 0}, {1, 2}, {2, 0}, {2, 2}};
  for (int op = 0; op < 9; ++op)
    for (uint32_t variant = 0; variant < 32; ++variant) {
      uint32_t mode = (variant & 1 ? GOC_ALU_NEG_A : 0) | (variant & 2 ? GOC_ALU_ABS_A : 0) |
                      ((variant >> 2 & 3) << 6) | (variant & 16 ? GOC_ALU_CLAMP : 0);
      double expected[32];
      for (int lane = 0; lane < 32; ++lane)
        expected[lane] = reference(op, inputs[lane], mode);
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
        // Identical output buffers retain only the high word, so compare that
        // alias case to a disjoint execution. A few allowed numeric ULPs can
        // straddle a high-word boundary in the higher-precision reference.
        uint32_t source[2][32], separate[2][32];
        uint32_t *pa[] = {source[0], source[1]}, *pd[] = {separate[0], separate[1]};
        for (int lane = 0; lane < 32; ++lane) {
          source[0][lane] = uint32_t(inputs[lane]);
          source[1][lane] = uint32_t(inputs[lane] >> 32);
        }
        ASSERT_EQ(functions[op](cpu, UINT32_MAX, mode, pd, pa, nullptr), GOC_SUCCESS);
        for (uint32_t exec_mask : rdna4_exec_masks())
          for (const auto &alias : aliases) {
            SCOPED_TRACE(::testing::Message() << op << "/" << mode << "/" << cpu << "/" << exec_mask
                                              << "/" << alias[0] << "/" << alias[1]);
            uint32_t storage[4][34], before[2][32], *v[4];
            for (int reg = 0; reg < 4; ++reg) {
              std::fill(storage[reg], storage[reg] + 34, 0xdeadbeef);
              v[reg] = storage[reg] + 1;
              if (reg < 2)
                for (int lane = 0; lane < 32; ++lane)
                  v[reg][lane] = uint32_t(inputs[lane] >> (32 * reg));
            }
            uint32_t *d[] = {v[alias[0]], v[alias[1]]};
            for (int reg = 0; reg < 2; ++reg)
              std::copy(d[reg], d[reg] + 32, before[reg]);
            ASSERT_EQ(functions[op](cpu, exec_mask, mode, d, v, nullptr), GOC_SUCCESS);
            for (int lane = 0; lane < 32; ++lane) {
              if (!((exec_mask >> lane) & 1)) {
                EXPECT_EQ(d[0][lane], before[0][lane]);
                EXPECT_EQ(d[1][lane], before[1][lane]);
                continue;
              }
              uint64_t got = d[0][lane] | (uint64_t(d[1][lane]) << 32);
              if (std::isnan(expected[lane])) {
                EXPECT_TRUE(std::isnan(number(got)));
              } else if (alias[0] == alias[1]) {
                EXPECT_EQ(d[0][lane], separate[1][lane]);
              } else if (op < 5 || expected[lane] == 0 || std::isinf(expected[lane])) {
                EXPECT_EQ(got, bits(expected[lane]));
              } else {
                EXPECT_DOUBLE_EQ(number(got), expected[lane]);
              }
            }
            for (const auto &reg : storage) {
              EXPECT_EQ(reg[0], 0xdeadbeef);
              EXPECT_EQ(reg[33], 0xdeadbeef);
            }
          }
      }
    }
}

TEST(Fp64Unary, LiteralBoundaries) {
  struct Case {
    int op;
    uint64_t input, want;
  };

  const Case cases[] = {
      {2, 0x3fe0000000000000, 0},
      {2, 0xbfe0000000000000, 0x8000000000000000},
      {2, 0x3ff8000000000000, 0x4000000000000000},
      {2, 0x4004000000000000, 0x4000000000000000},
      {2, 0xc004000000000000, 0xc000000000000000},
      {2, 0x400c000000000000, 0x4010000000000000},
      {2, 0x7ff0000000001234, 0x7ff8000000001234},
      {4, 0x8000000000000000, 0},
      {4, 0x8000000000000001, 0x3fefffffffffffff},
      {4, 1, 1},
      {4, 0xbfefffffffffffff, 0x3ca0000000000000},
      {5, 0x8000000000000000, 0x8000000000000000},
      {5, 1, 0x1e60000000000000},
      {5, 0x4010000000000000, 0x4000000000000000},
      {6, 0x7ff0000000000000, 0},
      {6, 0xfff0000000000000, 0x8000000000000000},
      {6, 0, 0x7ff0000000000000},
      {6, 0x8000000000000000, 0xfff0000000000000},
      {6, 0x0010000000000000, 0x7fd0000000000000},
      {7, 0x8000000000000000, 0xfff0000000000000},
      {7, 0x4010000000000000, 0x3fe0000000000000},
  };
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (const auto &test : cases) {
      uint32_t a[2][32], d[2][32];
      uint32_t *pa[] = {a[0], a[1]}, *pd[] = {d[0], d[1]};
      std::fill(a[0], a[0] + 32, uint32_t(test.input));
      std::fill(a[1], a[1] + 32, uint32_t(test.input >> 32));
      ASSERT_EQ(functions[test.op](cpu, UINT32_MAX, 0, pd, pa, nullptr), GOC_SUCCESS);
      for (int lane = 0; lane < 32; ++lane)
        EXPECT_EQ(d[0][lane] | (uint64_t(d[1][lane]) << 32), test.want)
            << cpu << "/" << test.op << "/" << test.input;
    }
}

TEST(Fp64Unary, Validation) {
  uint32_t a[2][32] = {}, d[2][32];
  uint32_t *pa[] = {a[0], a[1]}, *pd[] = {d[0], d[1]};
  for (auto fn : functions) {
    for (auto &reg : d)
      std::fill(reg, reg + 32, 0xdeadbeef);
    EXPECT_EQ(fn(0, UINT32_MAX, GOC_ALU_NEG_B, pd, pa, nullptr), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(fn(0, 0, GOC_ALU_HIGH_D, pd, pa, nullptr), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(
        fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, UINT32_MAX, 0, pd, pa, nullptr),
        GOC_ERROR_UNSUPPORTED_SEMANTICS);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL, 0U, 0, pd, pa, nullptr), GOC_SUCCESS);
    for (auto &reg : d)
      for (uint32_t value : reg)
        EXPECT_EQ(value, 0xdeadbeef);
  }
}

TEST(Fp64Unary, MantissaLiteralSubnormalsAndPassthrough) {
  const uint64_t cases[][2] = {{1, 0x3fe0000000000000},
                               {0x8000000000000001, 0xbfe0000000000000},
                               {0x000fffffffffffff, 0x3feffffffffffffe},
                               {0x0010000000000000, 0x3fe0000000000000},
                               {0x4018000000000000, 0x3fe8000000000000},
                               {0, 0},
                               {0x8000000000000000, 0x8000000000000000},
                               {0x7ff0000000000000, 0x7ff0000000000000},
                               {0xfff0000000000000, 0xfff0000000000000},
                               {0x7ff0000000001234, 0x7ff0000000001234},
                               {0xfff8000000001234, 0xfff8000000001234}};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (const auto &test : cases) {
      uint32_t a[2][32], d[2][32];
      uint32_t *pa[] = {a[0], a[1]}, *pd[] = {d[0], d[1]};
      std::fill(a[0], a[0] + 32, uint32_t(test[0]));
      std::fill(a[1], a[1] + 32, uint32_t(test[0] >> 32));
      ASSERT_EQ(goc_rdna4_v_frexp_mant_f64(cpu, UINT32_MAX, 0, pd, pa, nullptr), GOC_SUCCESS);
      for (int lane = 0; lane < 32; ++lane)
        EXPECT_EQ(d[0][lane] | (uint64_t(d[1][lane]) << 32), test[1]);
    }
}

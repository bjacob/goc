// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_exec_masks.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>

namespace {

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

int call(int op, uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
         const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (op == 0)
    return goc_rdna4_v_add_f64(flags, mask, mode, d, a, b);
  if (op == 1)
    return goc_rdna4_v_mul_f64(flags, mask, mode, d, a, b);
  return goc_rdna4_v_fma_f64(flags, mask, mode, d, a, b, c);
}

} // namespace

TEST(Fp64, AllModifiersMasksAndCrossHalfAliases) {
  const int aliases[][2] = {{6, 7}, {0, 1}, {2, 3}, {4, 5}, {1, 0},
                            {3, 2}, {5, 4}, {1, 2}, {4, 1}, {6, 6}};
  const double scales[] = {1, 2, 4, 0.5};
  for (int op = 0; op < 3; ++op)
    for (uint32_t mode = 0; mode < 512; ++mode) {
      if (op != 2 && (mode & (GOC_ALU_NEG_C | GOC_ALU_ABS_C)))
        continue;
      uint32_t input[6][32], expected[2][32];
      for (int lane = 0; lane < 32; ++lane) {
        long double x[] = {(lane - 17) * 0.25L, (lane % 7 - 3) * 0.5L, (lane % 11 - 5) * 0.25L};
        for (int operand = 0; operand < 3; ++operand) {
          uint64_t raw = bits(double(x[operand]));
          input[2 * operand][lane] = uint32_t(raw);
          input[2 * operand + 1][lane] = uint32_t(raw >> 32);
          if (mode & (GOC_ALU_ABS_A << operand))
            x[operand] = std::abs(x[operand]);
          if (mode & (GOC_ALU_NEG_A << operand))
            x[operand] = -x[operand];
        }
        // Small dyadic inputs make these long-double expressions exact.
        double want = double(op == 0 ? x[0] + x[1] : op == 1 ? x[0] * x[1] : x[0] * x[1] + x[2]);
        want *= scales[(mode >> 6) & 3];
        if (mode & GOC_ALU_CLAMP)
          want = !(want > 0) ? 0 : std::min(want, 1.0);
        expected[0][lane] = uint32_t(bits(want));
        expected[1][lane] = uint32_t(bits(want) >> 32);
      }
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (uint64_t mask : rdna4_exec_masks())
          for (const auto &alias : aliases) {
            SCOPED_TRACE(::testing::Message() << op << "/" << mode << "/" << cpu << "/" << mask
                                              << "/" << alias[0] << "/" << alias[1]);
            uint32_t storage[8][34], before[2][32];
            uint32_t *v[8];
            for (int reg = 0; reg < 8; ++reg) {
              std::fill(storage[reg], storage[reg] + 34, 0xdeadbeef);
              v[reg] = storage[reg] + 1;
              if (reg < 6)
                std::copy(input[reg], input[reg] + 32, v[reg]);
            }
            uint32_t *d[] = {v[alias[0]], v[alias[1]]};
            for (int reg = 0; reg < 2; ++reg)
              std::copy(d[reg], d[reg] + 32, before[reg]);
            ASSERT_EQ(call(op, cpu, mask, mode, d, v, v + 2, v + 4), GOC_SUCCESS);
            for (int reg = 0; reg < 2; ++reg)
              for (int lane = 0; lane < 32; ++lane) {
                // When D halves share a buffer, the second register's write wins.
                uint32_t want = (mask >> lane) & 1 ? expected[alias[0] == alias[1] ? 1 : reg][lane]
                                                   : before[reg][lane];
                EXPECT_EQ(d[reg][lane], want);
              }
            for (const auto &reg : storage) {
              EXPECT_EQ(reg[0], 0xdeadbeef);
              EXPECT_EQ(reg[33], 0xdeadbeef);
            }
          }
    }
}

TEST(Fp64, LiteralRoundingAndExceptionalValues) {
  struct Case {
    int op;
    uint64_t a, b, c, want;
    uint32_t mode;
  };

  const Case cases[] = {
      {0, 0x3ff0000000000000, 0x3ca0000000000000, 0, 0x3ff0000000000000, 0},
      {0, 0x8000000000000000, 0x8000000000000000, 0, 0x8000000000000000, 0},
      {0, 1, 1, 0, 2, 0},
      {0, 0x7ff0000000000000, 0xfff0000000000000, 0, 0x7ff8000000000000, 0},
      {0, 0x7ff0000000000001, 0, 0, 0, GOC_ALU_CLAMP},
      {1, 0x3ff0000000000001, 0x3feffffffffffffe, 0, 0x3ff0000000000000, 0},
      {1, 0x8000000000000000, 0x3ff0000000000000, 0, 0x8000000000000000, 0},
      {1, 0x8000000000000001, 1, 0, 0x8000000000000000, 0},
      {1, 0x0010000000000000, 0x3fe0000000000000, 0, 0x0008000000000000, 0},
      {1, 0x7fefffffffffffff, 0x4000000000000000, 0, 0x7ff0000000000000, 0},
      {2, 0x3ff0000000000001, 0x3feffffffffffffe, 0xbff0000000000000, 0xb970000000000000, 0},
      {2, 1, 0x3ff0000000000000, 1, 2, 0},
      {2, 0x8000000000000000, 0x3ff0000000000000, 0x8000000000000000, 0x8000000000000000, 0},
      {2, 0, 0x3ff0000000000000, 0x8000000000000000, 0, 0},
      {2, 0x7ff0000000000000, 0, 0, 0, GOC_ALU_CLAMP},
      {2, 0x3ff0000000000001, 0x3feffffffffffffe, 0xbff0000000000000, 0xb980000000000000,
       GOC_ALU_OMOD_2},
  };
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (const auto &test : cases) {
      uint32_t storage[8][32], *v[8];
      uint64_t raw[] = {test.a, test.b, test.c};
      for (int reg = 0; reg < 8; ++reg) {
        v[reg] = storage[reg];
        if (reg < 6)
          std::fill(v[reg], v[reg] + 32, uint32_t(raw[reg / 2] >> (32 * (reg % 2))));
      }
      ASSERT_EQ(call(test.op, cpu, UINT32_MAX, test.mode, v + 6, v, v + 2, v + 4), GOC_SUCCESS);
      for (int lane = 0; lane < 32; ++lane) {
        uint64_t got = v[6][lane] | (uint64_t(v[7][lane]) << 32);
        if (std::isnan(number(test.want)))
          EXPECT_TRUE(std::isnan(number(got)));
        else
          EXPECT_EQ(got, test.want) << cpu << "/" << test.op << "/" << test.a;
      }
    }
}

TEST(Fp64, Validation) {
  uint32_t data[2][32] = {}, output[2][32];
  uint32_t *a[] = {data[0], data[1]}, *d[] = {output[0], output[1]};
  for (int op = 0; op < 3; ++op) {
    for (auto &reg : output)
      std::fill(reg, reg + 32, 0xdeadbeef);
    EXPECT_EQ(call(op, 0, UINT32_MAX, GOC_ALU_HIGH_C, d, a, a, a), GOC_ERROR_INVALID_FLAGS);
    if (op < 2) {
      EXPECT_EQ(call(op, 0, UINT32_MAX, GOC_ALU_NEG_C, d, a, a, a), GOC_ERROR_INVALID_FLAGS);
    }
    EXPECT_EQ(
        call(op, GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, UINT32_MAX, 0, d, a, a, a),
        GOC_ERROR_UNSUPPORTED_SEMANTICS);
    EXPECT_EQ(call(op, GOC_SEMANTICS_EXACT_EMPIRICAL, 0xffffffff00000000, 0, d, a, a, a),
              GOC_SUCCESS);
    for (auto &reg : output)
      for (uint32_t value : reg)
        EXPECT_EQ(value, 0xdeadbeef);
  }
}

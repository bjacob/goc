// SPDX-License-Identifier: MIT

#include "exec_masks.h"
#include "goc/goc.h"
#include "omod_reference.h"

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

int call(int op, uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
         const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (op == 0)
    return goc_v_add_f64_rdna4(flags, exec_mask, mode, d, a, b, nullptr);
  if (op == 1)
    return goc_v_mul_f64_rdna4(flags, exec_mask, mode, d, a, b, nullptr);
  if (op == 3)
    return goc_v_min_num_f64(flags, exec_mask, mode, d, a, b, nullptr);
  if (op == 4)
    return goc_v_max_num_f64(flags, exec_mask, mode, d, a, b, nullptr);
  if (op == 5)
    return goc_v_minimum_f64(flags, exec_mask, mode, d, a, b, nullptr);
  if (op == 6)
    return goc_v_maximum_f64(flags, exec_mask, mode, d, a, b, nullptr);
  return goc_v_fma_f64_rdna4(flags, exec_mask, mode, d, a, b, c, nullptr);
}

int call_wave64(int op, uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
                uint32_t *excp_flag_user = nullptr) {
  if (op == 7)
    return goc_v_min_f64_wave64(flags, exec_mask, mode, d, a, b, excp_flag_user);
  if (op == 8)
    return goc_v_max_f64_wave64(flags, exec_mask, mode, d, a, b, excp_flag_user);
  if (op == 0)
    return goc_v_add_f64_rdna4_wave64(flags, exec_mask, mode, d, a, b, excp_flag_user);
  if (op == 1)
    return goc_v_mul_f64_rdna4_wave64(flags, exec_mask, mode, d, a, b, excp_flag_user);
  if (op == 3)
    return goc_v_min_num_f64_wave64(flags, exec_mask, mode, d, a, b, excp_flag_user);
  if (op == 4)
    return goc_v_max_num_f64_wave64(flags, exec_mask, mode, d, a, b, excp_flag_user);
  if (op == 5)
    return goc_v_minimum_f64_wave64(flags, exec_mask, mode, d, a, b, excp_flag_user);
  if (op == 6)
    return goc_v_maximum_f64_wave64(flags, exec_mask, mode, d, a, b, excp_flag_user);
  return goc_v_fma_f64_rdna4_wave64(flags, exec_mask, mode, d, a, b, c, excp_flag_user);
}

} // namespace

TEST(Fp64, AllModifiersMasksAndCrossHalfAliases) {
  const int aliases[][2] = {{6, 7}, {0, 1}, {2, 3}, {4, 5}, {1, 0},
                            {3, 2}, {5, 4}, {1, 2}, {4, 1}, {6, 6}};
  for (int op = 0; op < 7; ++op)
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
        if (op >= 3) {
          bool maximum = op == 4 || op == 6;
          if (x[0] == 0 && x[1] == 0)
            want = (maximum ? std::signbit(x[0]) && std::signbit(x[1])
                            : std::signbit(x[0]) || std::signbit(x[1]))
                       ? -0.0
                       : 0.0;
          else
            want = double(maximum ? std::max(x[0], x[1]) : std::min(x[0], x[1]));
        }
        want = goc_test::omod_f64_reference(want, mode);
        if (mode & GOC_ALU_CLAMP)
          want = !(want > 0) ? 0 : std::min(want, 1.0);
        expected[0][lane] = uint32_t(bits(want));
        expected[1][lane] = uint32_t(bits(want) >> 32);
      }
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (uint32_t exec_mask : exec_masks())
          for (const auto &alias : aliases) {
            SCOPED_TRACE(::testing::Message() << op << "/" << mode << "/" << cpu << "/" << exec_mask
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
            ASSERT_EQ(call(op, cpu, exec_mask, mode, d, v, v + 2, v + 4), GOC_SUCCESS);
            for (int reg = 0; reg < 2; ++reg)
              for (int lane = 0; lane < 32; ++lane) {
                // When D halves share a buffer, the second register's write wins.
                uint32_t want = (exec_mask >> lane) & 1
                                    ? expected[alias[0] == alias[1] ? 1 : reg][lane]
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
  for (int op = 0; op < 7; ++op) {
    for (auto &reg : output)
      std::fill(reg, reg + 32, 0xdeadbeef);
    EXPECT_EQ(call(op, 0, UINT32_MAX, GOC_ALU_HIGH_C, d, a, a, a), GOC_ERROR_INVALID_FLAGS);
    if (op != 2) {
      EXPECT_EQ(call(op, 0, UINT32_MAX, GOC_ALU_NEG_C, d, a, a, a), GOC_ERROR_INVALID_FLAGS);
    }
    EXPECT_EQ(
        call(op, GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, UINT32_MAX, 0, d, a, a, a),
        GOC_ERROR_UNSUPPORTED_SEMANTICS);
    EXPECT_EQ(call(op, GOC_SEMANTICS_EXACT_EMPIRICAL, 0U, 0, d, a, a, a), GOC_SUCCESS);
    for (auto &reg : output)
      for (uint32_t value : reg)
        EXPECT_EQ(value, 0xdeadbeef);
  }
}

namespace {

uint64_t minmax_reference(int op, uint64_t a, uint64_t b, uint32_t mode) {
  uint64_t operands[] = {a, b};
  for (int source = 0; source < 2; ++source) {
    if (mode & (GOC_ALU_ABS_A << source))
      operands[source] &= 0x7fffffffffffffffULL;
    if (mode & (GOC_ALU_NEG_A << source))
      operands[source] ^= 0x8000000000000000ULL;
  }
  const auto nan = [](uint64_t x) { return (x & 0x7fffffffffffffffULL) > 0x7ff0000000000000ULL; };
  uint64_t selected = 0;
  if (nan(operands[0]) || nan(operands[1])) {
    if (op < 2) {
      selected = nan(operands[0]) ? operands[1] : operands[0];
      if (nan(operands[0]) && nan(operands[1]))
        selected = operands[0];
    } else {
      // First signaling NaN, otherwise first quiet NaN.
      bool found = false;
      for (uint64_t x : operands)
        if (!found && nan(x) && !(x & 0x0008000000000000ULL)) {
          selected = x;
          found = true;
        }
      if (!found)
        selected = nan(operands[0]) ? operands[0] : operands[1];
    }
    if (nan(selected))
      selected |= 0x0008000000000000ULL;
  } else {
    const auto key = [](uint64_t x) {
      return x & 0x8000000000000000ULL ? ~x : x ^ 0x8000000000000000ULL;
    };
    std::sort(operands, operands + 2, [&](uint64_t x, uint64_t y) { return key(x) < key(y); });
    selected = operands[op % 2];
  }
  if (!nan(selected))
    selected = bits(goc_test::omod_f64_reference(number(selected), mode));
  if (mode & GOC_ALU_CLAMP) {
    if (nan(selected) || (selected & 0x8000000000000000ULL))
      selected = 0;
    else if (selected > 0x3ff0000000000000ULL)
      selected = 0x3ff0000000000000ULL;
  }
  return selected;
}

} // namespace

TEST(Fp64, MinMaxSpecialPairsAndAllModifiers) {
  const uint64_t values[] = {0,
                             0x8000000000000000,
                             1,
                             0x8000000000000001,
                             0x000fffffffffffff,
                             0x0010000000000000,
                             0x3fe0000000000000,
                             0xbfe0000000000000,
                             0x3ff0000000000000,
                             0xbff0000000000000,
                             0x3ff0000000000001,
                             0x3fefffffffffffff,
                             0x7fefffffffffffff,
                             0xffefffffffffffff,
                             0x7ff0000000000000,
                             0xfff0000000000000,
                             0x7ff8000000001234,
                             0xfff8000000005678,
                             0x7ff0000000001234,
                             0xfff0000000005678,
                             0x3fd0000000000000,
                             0x4000000000000000,
                             0xc008000000000000,
                             0x800fffffffffffff};
  for (int op = 0; op < 4; ++op)
    for (uint32_t variant = 0; variant < 128; ++variant) {
      uint32_t mode = (variant & 3) | ((variant & 12) << 1) | ((variant & 112) << 2);
      for (int batch = 0; batch < 18; ++batch) {
        uint32_t a[2][32], b[2][32], d[2][32];
        uint64_t expected[32];
        for (int lane = 0; lane < 32; ++lane) {
          uint64_t x = values[(batch * 32 + lane) / 24], y = values[(batch * 32 + lane) % 24];
          a[0][lane] = uint32_t(x);
          a[1][lane] = uint32_t(x >> 32);
          b[0][lane] = uint32_t(y);
          b[1][lane] = uint32_t(y >> 32);
          expected[lane] = minmax_reference(op, x, y, mode);
        }
        uint32_t *pa[] = {a[0], a[1]}, *pb[] = {b[0], b[1]}, *pd[] = {d[0], d[1]};
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
          SCOPED_TRACE(::testing::Message() << op << "/" << mode << "/" << batch << "/" << cpu);
          ASSERT_EQ(call(op + 3, cpu, UINT32_MAX, mode, pd, pa, pb, nullptr), GOC_SUCCESS);
          for (int lane = 0; lane < 32; ++lane)
            EXPECT_EQ(d[0][lane] | (uint64_t(d[1][lane]) << 32), expected[lane]);
        }
      }
    }
}

TEST(Fp64, MinMaxLiteralNaNPriorityAndZeros) {
  const uint64_t cases[][6] = {
      {0, 0x8000000000000000, 0x8000000000000000, 0, 0x8000000000000000, 0},
      {0x8000000000000000, 0, 0x8000000000000000, 0, 0x8000000000000000, 0},
      {0x7ff0000000001234, 0xbff0000000000000, 0xbff0000000000000, 0xbff0000000000000,
       0x7ff8000000001234, 0x7ff8000000001234},
      {0x7ff8000000001234, 0xfff0000000005678, 0x7ff8000000001234, 0x7ff8000000001234,
       0xfff8000000005678, 0xfff8000000005678},
      {0xfff0000000001234, 0x7ff0000000005678, 0xfff8000000001234, 0xfff8000000001234,
       0xfff8000000001234, 0xfff8000000001234},
  };
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (int op = 0; op < 4; ++op)
      for (const auto &test : cases) {
        uint32_t a[2][32], b[2][32], d[2][32];
        uint32_t *pa[] = {a[0], a[1]}, *pb[] = {b[0], b[1]}, *pd[] = {d[0], d[1]};
        for (int reg = 0; reg < 2; ++reg) {
          std::fill(a[reg], a[reg] + 32, uint32_t(test[0] >> (32 * reg)));
          std::fill(b[reg], b[reg] + 32, uint32_t(test[1] >> (32 * reg)));
        }
        ASSERT_EQ(call(op + 3, cpu, UINT32_MAX, 0, pd, pa, pb, nullptr), GOC_SUCCESS);
        for (int lane = 0; lane < 32; ++lane)
          EXPECT_EQ(d[0][lane] | (uint64_t(d[1][lane]) << 32), test[op + 2]);
      }
}

TEST(Fp64, MinmaxNaNPriorityAcrossPrecisions) {
  using Fn = decltype(&goc_v_min_num_f32);
  const Fn instructions[2][4] = {
      {goc_v_min_num_f32, goc_v_max_num_f32, goc_v_minimum_f32, goc_v_maximum_f32},
      {goc_v_min_num_f64, goc_v_max_num_f64, goc_v_minimum_f64, goc_v_maximum_f64}};
  // A, B, expected number result, expected propagating result. Payloads and
  // signs remain observable; signaling NaNs take precedence only when propagating.
  const uint64_t cases[2][4][4] = {
      {{0x7f800001, 0x7fc00002, 0x7fc00001, 0x7fc00001},
       {0x7fc00001, 0x7f800002, 0x7fc00001, 0x7fc00002},
       {0xffc00001, 0x3f800000, 0x3f800000, 0xffc00001},
       {0x3f800000, 0xff800002, 0x3f800000, 0xffc00002}},
      {{0x7ff0000000000001, 0x7ff8000000000002, 0x7ff8000000000001, 0x7ff8000000000001},
       {0x7ff8000000000001, 0x7ff0000000000002, 0x7ff8000000000001, 0x7ff8000000000002},
       {0xfff8000000000001, 0x3ff0000000000000, 0x3ff0000000000000, 0xfff8000000000001},
       {0x3ff0000000000000, 0xfff0000000000002, 0x3ff0000000000000, 0xfff8000000000002}}};
  for (unsigned format = 0; format < 2; ++format)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned op = 0; op < 4; ++op)
        for (const auto &sample : cases[format]) {
          uint32_t a[2][32], b[2][32], output[2][32] = {};
          for (unsigned reg = 0; reg < 2; ++reg)
            for (unsigned lane = 0; lane < 32; ++lane) {
              a[reg][lane] = uint32_t(sample[0] >> (32 * reg));
              b[reg][lane] = uint32_t(sample[1] >> (32 * reg));
            }
          const uint32_t *pa[] = {a[0], a[1]}, *pb[] = {b[0], b[1]};
          uint32_t *pd[] = {output[0], output[1]};
          ASSERT_EQ(instructions[format][op](cpu, UINT32_MAX, 0, pd, pa, pb, nullptr), GOC_SUCCESS);
          for (unsigned lane = 0; lane < 32; ++lane) {
            uint64_t actual = output[0][lane] | (uint64_t(output[1][lane]) << 32);
            EXPECT_EQ(actual, sample[op < 2 ? 2 : 3]) << format << '/' << cpu << '/' << op;
          }
        }
}

TEST(Fp64, Wave64AllModifiersMasksAndCrossWordAliases) {
  const int aliases[][2] = {{6, 7}, {0, 1}, {2, 3}, {4, 5}, {1, 0},
                            {3, 2}, {5, 4}, {1, 2}, {4, 1}, {6, 6}};
  for (int op = 0; op < 9; ++op)
    for (uint32_t mode = 0; mode < 512; ++mode) {
      if (op != 2 && (mode & (GOC_ALU_NEG_C | GOC_ALU_ABS_C)))
        continue;
      uint32_t input[6][64], expected[2][64];
      for (int lane = 0; lane < 64; ++lane) {
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
        if (op >= 3) {
          bool maximum = op == 4 || op == 6 || op == 8;
          if (x[0] == 0 && x[1] == 0)
            want = (maximum ? std::signbit(x[0]) && std::signbit(x[1])
                            : std::signbit(x[0]) || std::signbit(x[1]))
                       ? -0.0
                       : 0.0;
          else
            want = double(maximum ? std::max(x[0], x[1]) : std::min(x[0], x[1]));
        }
        want = goc_test::omod_f64_reference(want, mode);
        if (mode & GOC_ALU_CLAMP)
          want = !(want > 0) ? 0 : std::min(want, 1.0);
        expected[0][lane] = uint32_t(bits(want));
        expected[1][lane] = uint32_t(bits(want) >> 32);
      }
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (uint64_t exec_mask :
             {UINT64_MAX, uint64_t(0xffffffff00000000ULL), uint64_t(0x87654321fedcba98ULL)})
          for (const auto &alias : aliases) {
            SCOPED_TRACE(::testing::Message() << op << "/" << mode << "/" << cpu << "/" << exec_mask
                                              << "/" << alias[0] << "/" << alias[1]);
            uint32_t storage[8][66], before[2][64];
            uint32_t *v[8];
            for (int reg = 0; reg < 8; ++reg) {
              std::fill(storage[reg], storage[reg] + 66, 0xdeadbeef);
              v[reg] = storage[reg] + 1;
              if (reg < 6)
                std::copy(input[reg], input[reg] + 64, v[reg]);
            }
            uint32_t *d[] = {v[alias[0]], v[alias[1]]};
            for (int reg = 0; reg < 2; ++reg)
              std::copy(d[reg], d[reg] + 64, before[reg]);
            const uint32_t *inputs[] = {v[0], v[1], v[2], v[3], v[4], v[5]};
            uint32_t exceptions = 0x80000055;
            ASSERT_EQ(call_wave64(op, cpu, exec_mask, mode, d, inputs, inputs + 2,
                                  op == 2 ? inputs + 4 : nullptr, &exceptions),
                      GOC_SUCCESS);
            EXPECT_EQ(exceptions, 0x80000055U);
            for (int reg = 0; reg < 2; ++reg)
              for (int lane = 0; lane < 64; ++lane) {
                // When D halves share a buffer, the second register's write wins.
                uint32_t want = (exec_mask >> lane) & 1
                                    ? expected[alias[0] == alias[1] ? 1 : reg][lane]
                                    : before[reg][lane];
                EXPECT_EQ(d[reg][lane], want);
              }
            for (const auto &reg : storage) {
              EXPECT_EQ(reg[0], 0xdeadbeef);
              EXPECT_EQ(reg[65], 0xdeadbeef);
            }
          }
    }
}

TEST(Fp64, Wave64LiteralRoundingAndExceptionalValues) {
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
      uint32_t storage[8][64], *v[8];
      uint64_t raw[] = {test.a, test.b, test.c};
      for (int reg = 0; reg < 8; ++reg) {
        v[reg] = storage[reg];
        if (reg < 6)
          std::fill(v[reg], v[reg] + 64, uint32_t(raw[reg / 2] >> (32 * (reg % 2))));
      }
      ASSERT_EQ(call_wave64(test.op, cpu, UINT64_MAX, test.mode, v + 6, v, v + 2, v + 4),
                GOC_SUCCESS);
      for (int lane = 0; lane < 64; ++lane) {
        uint64_t got = v[6][lane] | (uint64_t(v[7][lane]) << 32);
        if (std::isnan(number(test.want)))
          EXPECT_TRUE(std::isnan(number(got)));
        else
          EXPECT_EQ(got, test.want) << cpu << "/" << test.op << "/" << test.a;
      }
    }
}

TEST(Fp64, Wave64ReportingAndValidation) {
  for (unsigned op = 0; op < 9; ++op) {
    EXPECT_EQ(call_wave64(op, 0, 0, 0, nullptr, nullptr, nullptr, nullptr), GOC_SUCCESS);
    EXPECT_EQ(call_wave64(op, 0, UINT64_MAX, 1ULL << 32, nullptr, nullptr, nullptr, nullptr),
              GOC_ERROR_INVALID_FLAGS);
    uint32_t exceptions = 0x80000055;
    EXPECT_EQ(call_wave64(op, GOC_SEMANTICS_EXACT_EMPIRICAL, UINT64_MAX, 0, nullptr, nullptr,
                          nullptr, nullptr, &exceptions),
              GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
    EXPECT_EQ(exceptions, 0x80000055U);
  }
}

TEST(Fp64, ArchitectureScalingAndGuestDenormalsAcrossAliases) {
  const decltype(&goc_v_add_f64) narrow[][2] = {{goc_v_add_f64, goc_v_mul_f64},
                                                {goc_v_add_f64_rdna4, goc_v_mul_f64_rdna4}};
  const decltype(&goc_v_add_f64_wave64) wide[][2] = {
      {goc_v_add_f64_wave64, goc_v_mul_f64_wave64},
      {goc_v_add_f64_rdna4_wave64, goc_v_mul_f64_rdna4_wave64}};
  const decltype(&goc_v_fma_f64) fma[] = {goc_v_fma_f64, goc_v_fma_f64_rdna4};
  const decltype(&goc_v_fma_f64_wave64) fma_wide[] = {goc_v_fma_f64_wave64,
                                                      goc_v_fma_f64_rdna4_wave64};
  const uint64_t values[] = {0,
                             0x8000000000000000ULL,
                             1,
                             0x8000000000000001ULL,
                             0x000fffffffffffffULL,
                             0x800fffffffffffffULL,
                             0x0010000000000000ULL,
                             0x8010000000000000ULL,
                             0x3fe0000000000000ULL,
                             0x3ff0000000000000ULL,
                             0xc000000000000000ULL,
                             0x7fefffffffffffffULL,
                             0x7ff0000000000000ULL,
                             0x7ff8000000000001ULL};
  const unsigned aliases[][2] = {{6, 7}, {0, 1}, {1, 0}, {3, 2}, {4, 5}, {1, 2}, {6, 6}};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned arch = 0; arch < 2; ++arch)
      for (unsigned op = 0; op < 3; ++op)
        for (unsigned settings = 0; settings < 8; ++settings)
          for (unsigned variant = 0; variant < 16; ++variant)
            for (unsigned width : {32U, 64U})
              for (const auto &alias : aliases) {
                uint64_t flags = cpu | (settings & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                                 (settings & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) |
                                 (settings & 4 ? GOC_FP_IEEE_MODE : 0);
                uint32_t mode = ((variant & 3) << 6) |
                                (variant & 4 ? GOC_ALU_NEG_A | GOC_ALU_ABS_B : 0) |
                                (variant & 8 ? GOC_ALU_CLAMP : 0);
                if (op == 2 && (variant & 4))
                  mode |= GOC_ALU_NEG_C;
                uint32_t words[8][66], before[8][66];
                for (unsigned reg = 0; reg < 8; ++reg)
                  for (unsigned lane = 0; lane < 66; ++lane) {
                    uint64_t raw = values[(lane * (reg / 2 + 1) + reg / 2 * 3) % 14];
                    words[reg][lane] = uint32_t(raw >> (32 * (reg & 1)));
                  }
                std::memcpy(before, words, sizeof words);
                uint32_t *d[] = {words[alias[0]] + 1, words[alias[1]] + 1};
                const uint32_t *a[] = {words[0] + 1, words[1] + 1};
                const uint32_t *b[] = {words[2] + 1, words[3] + 1};
                const uint32_t *c[] = {words[4] + 1, words[5] + 1};
                uint64_t exec_mask = 0x87654321fedcba98ULL;
                uint32_t state = 0x12345678;
                int status;
                if (width == 32)
                  status =
                      op == 2 ? fma[arch](flags, uint32_t(exec_mask), mode, d, a, b, c, &state)
                              : narrow[arch][op](flags, uint32_t(exec_mask), mode, d, a, b, &state);
                else
                  status = op == 2 ? fma_wide[arch](flags, exec_mask, mode, d, a, b, c, &state)
                                   : wide[arch][op](flags, exec_mask, mode, d, a, b, &state);
                ASSERT_EQ(status, GOC_SUCCESS);
                EXPECT_EQ(state, 0x12345678U);
                for (unsigned lane = 0; lane < width; ++lane) {
                  double input[3];
                  for (unsigned operand = 0; operand < 3; ++operand) {
                    uint64_t raw = before[2 * operand][lane + 1] |
                                   (uint64_t(before[2 * operand + 1][lane + 1]) << 32);
                    if ((settings & 1) && (raw & 0x7fffffffffffffffULL) < 0x0010000000000000ULL)
                      raw &= 0x8000000000000000ULL;
                    if (mode & (GOC_ALU_ABS_A << operand))
                      raw &= 0x7fffffffffffffffULL;
                    if (mode & (GOC_ALU_NEG_A << operand))
                      raw ^= 0x8000000000000000ULL;
                    input[operand] = number(raw);
                  }
                  double result = op == 0   ? input[0] + input[1]
                                  : op == 1 ? input[0] * input[1]
                                            : std::fma(input[0], input[1], input[2]);
                  uint32_t effective = !arch && (!(settings & 2) || (settings & 4))
                                           ? mode & ~GOC_ALU_OMOD_HALF
                                           : mode;
                  result = goc_test::omod_f64_reference(result, effective);
                  if (mode & GOC_ALU_CLAMP)
                    result = !(result > 0) ? 0 : std::min(result, 1.0);
                  uint64_t expected = bits(result);
                  if ((settings & 2) && (expected & 0x7fffffffffffffffULL) < 0x0010000000000000ULL)
                    expected &= 0x8000000000000000ULL;
                  for (unsigned word = 0; word < 2; ++word) {
                    if (!((exec_mask >> lane) & 1)) {
                      EXPECT_EQ(d[word][lane], before[alias[word]][lane + 1]);
                      continue;
                    }
                    // Duplicate destination buffers retain the high-word store.
                    unsigned output_word = alias[0] == alias[1] ? 1 : word;
                    if (std::isnan(result)) {
                      if (output_word == 1) {
                        EXPECT_GT(d[word][lane] & 0x7fffffffU, 0x7ff00000U);
                      }
                    } else {
                      EXPECT_EQ(d[word][lane], uint32_t(expected >> (32 * output_word)))
                          << cpu << '/' << arch << '/' << op << '/' << settings << '/' << mode;
                    }
                  }
                }
                for (unsigned reg = 0; reg < 8; ++reg) {
                  EXPECT_EQ(words[reg][0], before[reg][0]);
                  EXPECT_EQ(words[reg][width + 1], before[reg][width + 1]);
                }
              }
}

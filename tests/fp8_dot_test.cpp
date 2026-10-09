// SPDX-License-Identifier: MIT

#include "exec_masks.h"
#include "goc/goc.h"
#include "internal.h"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <initializer_list>
#include <limits>
#include <random>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_v_mad_u32_u24);
const Fn functions[] = {goc_v_dot4_f32_fp8_fp8, goc_v_dot4_f32_fp8_bf8, goc_v_dot4_f32_bf8_fp8,
                        goc_v_dot4_f32_bf8_bf8};

// Independent mathematical decode; no implementation conversion helpers.
double decode(uint8_t value, bool bf8) {
  const int radix = bf8 ? 4 : 8, bias = bf8 ? 15 : 7;
  const int magnitude = value % 128, exponent = magnitude / radix, fraction = magnitude % radix;
  double result;
  if (bf8 && exponent == 31)
    result = fraction ? std::numeric_limits<double>::quiet_NaN()
                      : std::numeric_limits<double>::infinity();
  else if (!bf8 && magnitude == 127)
    result = std::numeric_limits<double>::quiet_NaN();
  else
    result = std::ldexp(exponent ? 1.0 + double(fraction) / radix : double(fraction) / radix,
                        (exponent ? exponent : 1) - bias);
  return value >= 128 ? -result : result;
}

void compare(uint32_t word, double expected) {
  float actual = goc::as_float(word), rounded = float(expected);
  if (std::isnan(rounded)) {
    EXPECT_TRUE(std::isnan(actual));
  } else {
    EXPECT_FLOAT_EQ(actual, rounded);
  }
}

} // namespace

TEST(Fp8Dot, ExhaustiveBytePairsAllFormatsAndModifiers) {
  for (int op = 0; op < 4; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint32_t modifiers : {0U, GOC_DOT_NEG_C, GOC_DOT_ABS_C, GOC_DOT_ABS_C | GOC_DOT_NEG_C})
        for (unsigned base = 0; base < 65536; base += 32) {
          SCOPED_TRACE(::testing::Message() << op << "/" << cpu << "/" << modifiers << "/" << base);
          uint32_t a[32], b[32], c[32], d[32];
          for (unsigned lane = 0; lane < 32; ++lane) {
            a[lane] = ((base + lane) / 256) * 0x01010101u;
            b[lane] = ((base + lane) % 256) * 0x01010101u;
            c[lane] = goc::as_bits(-1.25f);
          }
          auto pa = a, pb = b, pc = c, pd = d;
          ASSERT_EQ(functions[op](cpu, UINT32_MAX, modifiers, &pd, &pa, &pb, &pc), GOC_SUCCESS);
          double acc = modifiers & GOC_DOT_ABS_C ? 1.25 : -1.25;
          if (modifiers & GOC_DOT_NEG_C)
            acc = -acc;
          for (unsigned lane = 0; lane < 32; ++lane)
            compare(d[lane], 4 * decode(uint8_t((base + lane) / 256), op >= 2) *
                                     decode(uint8_t((base + lane) % 256), op & 1) +
                                 acc);
        }
}

TEST(Fp8Dot, MixedBytesMasksAndAliases) {
  std::mt19937 rng(54321);
  uint32_t inputs[3][32];
  for (auto &reg : inputs)
    for (auto &word : reg)
      word = rng();
  // Every finite integer factor in [-2,2] is exactly encoded in both formats.
  const uint8_t fp8[] = {0xc0, 0xb8, 0, 0x38, 0x40};
  const uint8_t bf8[] = {0xc0, 0xbc, 0, 0x3c, 0x40};
  for (int op = 0; op < 4; ++op) {
    int factors[2][32][4];
    for (int lane = 0; lane < 32; ++lane) {
      for (int operand = 0; operand < 2; ++operand) {
        inputs[operand][lane] = 0;
        const auto *codes = (operand ? op & 1 : op >= 2) ? bf8 : fp8;
        for (int byte = 0; byte < 4; ++byte) {
          int index = rng() % 5;
          factors[operand][lane][byte] = index - 2;
          inputs[operand][lane] |= uint32_t(codes[index]) << (8 * byte);
        }
      }
      inputs[2][lane] = goc::as_bits(float(lane - 16));
    }
    for (uint32_t mode : {0U, GOC_DOT_NEG_C, GOC_DOT_ABS_C, GOC_DOT_NEG_C | GOC_DOT_ABS_C})
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (uint32_t exec_mask : exec_masks())
          for (int alias = 0; alias < 4; ++alias) {
            uint32_t storage[4][34], before[32];
            uint32_t *v[4];
            for (int reg = 0; reg < 4; ++reg) {
              std::fill(storage[reg], storage[reg] + 34, 0xdeadbeef);
              v[reg] = storage[reg] + 1;
              if (reg < 3)
                std::copy(inputs[reg], inputs[reg] + 32, v[reg]);
            }
            std::copy(v[alias], v[alias] + 32, before);
            ASSERT_EQ(functions[op](cpu, exec_mask, mode, &v[alias], &v[0], &v[1], &v[2]),
                      GOC_SUCCESS);
            for (int lane = 0; lane < 32; ++lane) {
              int expected = lane - 16;
              if (mode & GOC_DOT_ABS_C)
                expected = std::abs(expected);
              if (mode & GOC_DOT_NEG_C)
                expected = -expected;
              for (int byte = 0; byte < 4; ++byte)
                expected += factors[0][lane][byte] * factors[1][lane][byte];
              if ((exec_mask >> lane) & 1) {
                EXPECT_FLOAT_EQ(goc::as_float(v[alias][lane]), float(expected));
              } else {
                EXPECT_EQ(v[alias][lane], before[lane]);
              }
            }
            for (const auto &reg : storage) {
              EXPECT_EQ(reg[0], 0xdeadbeef);
              EXPECT_EQ(reg[33], 0xdeadbeef);
            }
          }
  }
}

TEST(Fp8Dot, ValidationAndLooseFallback) {
  for (auto fn : functions) {
    uint32_t a[32] = {}, d[32];
    std::fill(d, d + 32, 0xdeadbeef);
    auto pa = a, pd = d;
    for (uint32_t exec_mask : {0U, UINT32_MAX}) {
      for (uint32_t invalid : {GOC_DOT_NEG_LO_A, GOC_DOT_CLAMP, GOC_DOT_LO_A_HIGH, 1U << 31})
        EXPECT_EQ(fn(0, exec_mask, invalid, &pd, &pa, &pa, &pa), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, exec_mask, 0, &pd, &pa,
                   &pa, &pa),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
    }
    for (auto word : d)
      EXPECT_EQ(word, 0xdeadbeef);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, 0, &pd, &pa, &pa, &pa), GOC_SUCCESS);
    for (auto word : d)
      EXPECT_EQ(word, 0u);
  }
}

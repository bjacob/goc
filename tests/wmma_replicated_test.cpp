// SPDX-License-Identifier: MIT

#include "fuzzy.h"
#include "goc/goc.h"
#include "loose_tolerances.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {

uint32_t as_bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

using Wmma = decltype(&goc_v_wmma_f32_16x16x16_f16);

Wmma instruction(bool bf16, int lanes, bool rdna4 = false) {
  if (rdna4) {
    if (lanes == 64)
      return bf16 ? goc_v_wmma_f32_16x16x16_bf16_rdna4_wave64
                  : goc_v_wmma_f32_16x16x16_f16_rdna4_wave64;
    return bf16 ? goc_v_wmma_f32_16x16x16_bf16_rdna4 : goc_v_wmma_f32_16x16x16_f16_rdna4;
  }
  if (lanes == 64)
    return bf16 ? goc_v_wmma_f32_16x16x16_bf16_wave64 : goc_v_wmma_f32_16x16x16_f16_wave64;
  return bf16 ? goc_v_wmma_f32_16x16x16_bf16 : goc_v_wmma_f32_16x16x16_f16;
}

uint16_t encode(int value, bool bf16) {
  const uint16_t half[] = {0xc000, 0xbc00, 0, 0x3c00, 0x4000};
  const uint16_t bfloat[] = {0xc000, 0xbf80, 0, 0x3f80, 0x4000};
  return bf16 ? bfloat[value + 2] : half[value + 2];
}

int a_value(int row, int k) { return (row * 3 + k * 2) % 5 - 2; }

int b_value(int k, int col) { return (k * 4 + col * 3) % 5 - 2; }

} // namespace

// Independent logical matrix reference; no production layout or arithmetic helpers.
TEST(WmmaReplicated, DenseMatricesModifiersAliasesAndGuards) {
  for (int lanes : {32, 64})
    for (bool bf16 : {false, true})
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (uint64_t mode = 0; mode < 64; ++mode)
          for (int dst : {0, 4, 8, 16, 24}) {
            SCOPED_TRACE(::testing::Message()
                         << lanes << "/" << bf16 << "/" << cpu << "/" << mode << "/" << dst);
            uint32_t storage[32][66], expected[32][66];
            uint32_t *v[32];
            for (int reg = 0; reg < 32; ++reg) {
              std::fill(storage[reg], storage[reg] + 66, 0xdeadbeef);
              v[reg] = storage[31 - reg] + 1;
            }
            for (int lane = 0; lane < lanes; ++lane) {
              for (int k = 0; k < 16; k += 2) {
                v[k / 2][lane] =
                    encode(a_value(lane % 16 + lane / 16, k), bf16) |
                    (uint32_t(encode(a_value(lane % 16 + lane / 16, k + 1), bf16)) << 16);
                v[8 + k / 2][lane] =
                    encode(b_value(k, lane % 16 + lane / 16), bf16) |
                    (uint32_t(encode(b_value(k + 1, lane % 16 + lane / 16), bf16)) << 16);
              }
              for (int reg = 0; reg < 256 / lanes; ++reg)
                v[16 + reg][lane] = as_bits(float(reg * 3 - lane));
            }
            std::memcpy(expected, storage, sizeof(storage));
            for (int reg = 0; reg < 256 / lanes; ++reg)
              for (int lane = 0; lane < lanes; ++lane) {
                int row = reg * (lanes / 16) + lane / 16, col = lane % 16;
                int sum = reg * 3 - lane;
                if ((mode & GOC_WMMA_ABS_C) && sum < 0)
                  sum = -sum;
                if (mode & GOC_WMMA_NEG_C)
                  sum = -sum;
                for (int k = 0; k < 16; ++k) {
                  int av = a_value(row + lane / 16, k), bv = b_value(k, col + lane / 16);
                  if (mode & (k % 2 ? GOC_WMMA_NEG_HI_A : GOC_WMMA_NEG_LO_A))
                    av = -av;
                  if (mode & (k % 2 ? GOC_WMMA_NEG_HI_B : GOC_WMMA_NEG_LO_B))
                    bv = -bv;
                  sum += av * bv;
                }
                expected[31 - dst - reg][lane + 1] = as_bits(float(sum));
              }
            ASSERT_EQ(instruction(bf16, lanes)(cpu, mode, v + dst, v, v + 8, v + 16), GOC_SUCCESS);
            EXPECT_EQ(std::memcmp(storage, expected, sizeof(storage)), 0);
          }
}

TEST(WmmaReplicated, SameMnemonicDifferentInputFootprint) {
  for (int lanes : {32, 64})
    for (bool bf16 : {false, true}) {
      uint32_t data[32][64] = {};
      uint32_t *v[32];
      for (int reg = 0; reg < 32; ++reg)
        v[reg] = data[reg];
      for (int reg = 0; reg < 8; ++reg)
        for (int lane = 0; lane < lanes; ++lane) {
          uint32_t a = encode(reg < 4 ? 1 : 2, bf16), b = encode(1, bf16);
          v[reg][lane] = a | (a << 16);
          v[8 + reg][lane] = b | (b << 16);
        }
      ASSERT_EQ(instruction(bf16, lanes)(0, 0, v + 24, v, v + 8, v + 16), GOC_SUCCESS);
      for (int reg = 0; reg < 256 / lanes; ++reg)
        for (int lane = 0; lane < lanes; ++lane)
          EXPECT_EQ(v[24 + reg][lane], as_bits(24.0f));
      ASSERT_EQ(instruction(bf16, lanes, true)(0, 0, v + 24, v, v + 8, v + 16), GOC_SUCCESS);
      for (int reg = 0; reg < 256 / lanes; ++reg)
        for (int lane = 0; lane < lanes; ++lane)
          EXPECT_EQ(v[24 + reg][lane], as_bits(16.0f));
    }
}

TEST(WmmaReplicated, ValidationAndLooseFallback) {
  for (int lanes : {32, 64})
    for (bool bf16 : {false, true}) {
      auto fn = instruction(bf16, lanes);
      uint32_t data[8][64] = {};
      uint32_t *v[8];
      for (int reg = 0; reg < 8; ++reg)
        v[reg] = data[reg];
      for (auto &reg : data)
        std::fill(reg, reg + 64, 0x12345678);
      uint32_t before[8][64];
      std::memcpy(before, data, sizeof(before));
      EXPECT_EQ(fn(GOC_SEMANTICS_MASK | GOC_SEMANTICS_STRICT, 0, v, nullptr, nullptr, nullptr),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
      EXPECT_EQ(std::memcmp(data, before, sizeof(before)), 0);
      EXPECT_EQ(fn(1ULL << 63, 0, v, nullptr, nullptr, nullptr), GOC_ERROR_INVALID_FLAGS);
      for (unsigned bit = 0; bit < 64; ++bit) {
        uint64_t mode = 1ULL << bit;
        if (mode & 63)
          continue;
        EXPECT_EQ(fn(0, mode, v, nullptr, nullptr, nullptr), GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(std::memcmp(data, before, sizeof(before)), 0);
      }
      std::memset(data, 0, sizeof(data));
      EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL, 0, v, v, v, v), GOC_SUCCESS);
      for (const auto &reg : data)
        for (uint32_t value : reg)
          EXPECT_EQ(value, 0U);
    }
}

TEST(WmmaReplicated, FractionalMatricesAgainstDoubleReference) {
  // Independent mathematical reference, not a model of empirical exact rounding.
  // RDNA3's replicas are intentionally distinct; each output must use its own
  // 16-lane group. RDNA4 distributes a single matrix across lane groups instead.
  for (bool rdna4 : {false, true})
    for (int width : {32, 64})
      for (bool bf16 : {false, true}) {
        std::mt19937 random(63827);
        for (unsigned batch = 0; batch < 64; ++batch) {
          uint16_t codes[4][2][16][16];
          double values[4][2][16][16], expected[16][16];
          unsigned mantissa = bf16 ? 7 : 10, bias = bf16 ? 127 : 15;
          for (int group = 0; group < (rdna4 ? 1 : width / 16); ++group)
            for (int operand = 0; operand < 2; ++operand)
              for (int index = 0; index < 16; ++index)
                for (int k = 0; k < 16; ++k) {
                  unsigned bits = random();
                  unsigned exponent = bias - 6 + (bits >> 16) % 8;
                  unsigned fraction = bits & ((1U << mantissa) - 1);
                  codes[group][operand][index][k] =
                      (bits & 32768) | (exponent << mantissa) | fraction;
                  double value = std::ldexp(double((1U << mantissa) + fraction),
                                            int(exponent) - int(bias) - int(mantissa));
                  values[group][operand][index][k] = bits & 32768 ? -value : value;
                }
          uint32_t original[32][66];
          for (auto &reg : original)
            std::fill(reg, reg + 66, 0xdeadbeef);
          for (int lane = 0; lane < width; ++lane)
            for (int reg = 0; reg < (rdna4 ? 128 / width : 8); ++reg)
              for (int operand = 0; operand < 2; ++operand) {
                int group = rdna4 ? 0 : lane / 16;
                int k = reg * 2 + (rdna4 ? (lane / 16) * (256 / width) : 0);
                original[31 - 8 * operand - reg][lane + 1] =
                    codes[group][operand][lane % 16][k] |
                    (uint32_t(codes[group][operand][lane % 16][k + 1]) << 16);
              }
          for (int row = 0; row < 16; ++row)
            for (int col = 0; col < 16; ++col) {
              int group = rdna4 ? 0 : row % (width / 16);
              int reg = rdna4 ? row % (256 / width) : row / (width / 16);
              int lane = rdna4 ? col + 16 * (row / 8) + (width == 64 ? 32 * (row / 4 % 2) : 0)
                               : col + 16 * group;
              float c = float(int(random() % 2000001) - 1000000) / 262144.0f;
              original[31 - 16 - reg][lane + 1] = as_bits(c);
              double sum = batch & GOC_WMMA_ABS_C ? std::abs(c) : c;
              if (batch & GOC_WMMA_NEG_C)
                sum = -sum;
              for (int k = 0; k < 16; ++k) {
                double a = values[group][0][row][k], b = values[group][1][col][k];
                if (batch & (k % 2 ? GOC_WMMA_NEG_HI_A : GOC_WMMA_NEG_LO_A))
                  a = -a;
                if (batch & (k % 2 ? GOC_WMMA_NEG_HI_B : GOC_WMMA_NEG_LO_B))
                  b = -b;
                sum += a * b;
              }
              expected[row][col] = sum;
            }
          const int destinations[] = {0, 4, 8, 16, 24};
          int dst = destinations[batch % 5];
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
            uint32_t storage[32][66], *v[32];
            std::memcpy(storage, original, sizeof(storage));
            for (int reg = 0; reg < 32; ++reg)
              v[reg] = storage[31 - reg] + 1;
            ASSERT_EQ(instruction(bf16, width, rdna4)(cpu, batch, v + dst, v, v + 8, v + 16),
                      GOC_SUCCESS);
            for (int row = 0; row < 16; ++row)
              for (int col = 0; col < 16; ++col) {
                int reg = rdna4 ? row % (256 / width) : row / (width / 16);
                int lane = rdna4 ? col + 16 * (row / 8) + (width == 64 ? 32 * (row / 4 % 2) : 0)
                                 : col + 16 * (row % (width / 16));
                float actual;
                std::memcpy(&actual, &v[dst + reg][lane], sizeof(actual));
                EXPECT_TRUE(
                    goc_test::fuzzy(actual, expected[row][col], goc_test::wmma_f32_tolerance))
                    << rdna4 << '/' << width << '/' << bf16 << '/' << batch << '/' << cpu
                    << " row=" << row << " col=" << col;
              }
            for (int reg = 0; reg < 32; ++reg)
              for (int word = 0; word < 66; ++word)
                if (reg < dst || reg >= dst + 256 / width || word == 0 || word > width) {
                  EXPECT_EQ(storage[31 - reg][word], original[31 - reg][word]);
                }
          }
        }
      }
}

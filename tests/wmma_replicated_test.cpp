// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "internal.h"

#include <algorithm>
#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>

namespace {

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
        for (uint64_t mode : {uint64_t(0), uint64_t(GOC_WMMA_NEG_C), uint64_t(GOC_WMMA_ABS_C),
                              uint64_t(GOC_WMMA_NEG_C | GOC_WMMA_ABS_C)})
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
                v[k / 2][lane] = encode(a_value(lane % 16, k), bf16) |
                                 (uint32_t(encode(a_value(lane % 16, k + 1), bf16)) << 16);
                v[8 + k / 2][lane] = encode(b_value(k, lane % 16), bf16) |
                                     (uint32_t(encode(b_value(k + 1, lane % 16), bf16)) << 16);
              }
              for (int reg = 0; reg < 256 / lanes; ++reg)
                v[16 + reg][lane] = goc::as_bits(float(reg * 3 - lane));
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
                for (int k = 0; k < 16; ++k)
                  sum += a_value(row, k) * b_value(k, col);
                expected[31 - dst - reg][lane + 1] = goc::as_bits(float(sum));
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
          EXPECT_EQ(v[24 + reg][lane], goc::as_bits(24.0f));
      ASSERT_EQ(instruction(bf16, lanes, true)(0, 0, v + 24, v, v + 8, v + 16), GOC_SUCCESS);
      for (int reg = 0; reg < 256 / lanes; ++reg)
        for (int lane = 0; lane < lanes; ++lane)
          EXPECT_EQ(v[24 + reg][lane], goc::as_bits(16.0f));
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
      data[0][0] = 0x12345678;
      EXPECT_EQ(
          fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, v, nullptr, nullptr, nullptr),
          GOC_ERROR_UNSUPPORTED_SEMANTICS);
      EXPECT_EQ(data[0][0], 0x12345678U);
      EXPECT_EQ(fn(1ULL << 63, 0, v, nullptr, nullptr, nullptr), GOC_ERROR_INVALID_FLAGS);
      for (unsigned bit = 0; bit < 64; ++bit) {
        uint64_t mode = 1ULL << bit;
        if (mode & (GOC_WMMA_NEG_C | GOC_WMMA_ABS_C))
          continue;
        EXPECT_EQ(fn(0, mode, v, nullptr, nullptr, nullptr), GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(data[0][0], 0x12345678U);
      }
      std::memset(data, 0, sizeof(data));
      EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL, 0, v, v, v, v), GOC_SUCCESS);
      for (const auto &reg : data)
        for (uint32_t value : reg)
          EXPECT_EQ(value, 0U);
    }
}

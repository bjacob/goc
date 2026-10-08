// SPDX-License-Identifier: MIT

#include "goc_common.h"
#include "goc_rdna4.h"
#include "internal.h"
#include "subbyte_golden.h"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <initializer_list>
#include <random>
#include <stdint.h>

namespace {

using Wmma = decltype(&goc_rdna4_v_wmma_f32_16x16x16_fp8_fp8);

const Wmma floating[] = {
    goc_rdna4_v_wmma_f32_16x16x16_fp8_fp8, goc_rdna4_v_wmma_f32_16x16x16_fp8_bf8,
    goc_rdna4_v_wmma_f32_16x16x16_bf8_fp8, goc_rdna4_v_wmma_f32_16x16x16_bf8_bf8};
const Wmma integer[] = {goc_rdna4_v_wmma_i32_16x16x16_iu8, goc_rdna4_v_wmma_i32_16x16x16_iu4,
                        goc_rdna4_v_wmma_i32_16x16x32_iu4};

struct Registers {
  uint32_t storage[24][35] = {};
  uint32_t *v[24];

  Registers() {
    for (int i = 0; i < 24; ++i) {
      storage[i][0] = storage[i][34] = 0xdeadbeef;
      v[i] = storage[23 - i] + 1;
    }
  }

  void guards() {
    for (auto &reg : storage) {
      EXPECT_EQ(reg[0], 0xdeadbeef);
      EXPECT_EQ(reg[34], 0xdeadbeef);
    }
  }
};

// Pack logical input rows (A) or columns (B) into wave32 VGPRs. Groups of
// eight K elements alternate between the low and high half-wave.
void set(uint32_t *const *v, int index, int k, int bits, uint32_t value) {
  int group = k / 8;
  int reg = (group / 2) * (bits / 4) + (k % 8) / (32 / bits);
  int lane = index + 16 * (group % 2), shift = (k % (32 / bits)) * bits;
  uint32_t mask = ((1u << bits) - 1) << shift;
  v[reg][lane] = (v[reg][lane] & ~mask) | (value << shift);
}

void check(Registers &r, int dst, uint64_t mask, const uint32_t *golden,
           const uint32_t (&before)[8][32]) {
  for (int row = 0; row < 16; ++row)
    for (int col = 0; col < 16; ++col) {
      int lane = col + 16 * (row / 8), reg = row % 8;
      EXPECT_EQ(r.v[dst + reg][lane],
                ((mask >> lane) & 1) ? golden[row * 16 + col] : before[reg][lane])
          << "row=" << row << " col=" << col;
    }
  r.guards();
}

void save(Registers &r, int dst, uint32_t (&before)[8][32]) {
  for (int reg = 0; reg < 8; ++reg)
    std::copy(r.v[dst + reg], r.v[dst + reg] + 32, before[reg]);
}

} // namespace

TEST(SubbyteWmma, Fp8DenseGoldensMasksAndOverlap) {
  for (int format = 0; format < 4; ++format)
    for (int dst : {0, 4, 8, 16})
      for (uint64_t mask :
           {UINT64_C(0), UINT64_C(0xa55a1234), UINT64_C(0xffffffff00000000), UINT64_MAX})
        for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL}) {
          Registers r;
          std::minstd_rand random(12056925);
          for (int operand = 0; operand < 2; ++operand) {
            bool bf8 = operand ? format & 1 : format & 2;
            for (int i = 0; i < 256; ++i) {
              uint32_t x = random();
              uint32_t code = (x & 128) | ((bf8 ? 48 : 32) + x % (bf8 ? 24 : 48));
              set(r.v + 4 * operand, operand ? i % 16 : i / 16, operand ? i / 16 : i % 16, 8, code);
            }
          }
          for (int row = 0; row < 16; ++row)
            for (int col = 0; col < 16; ++col)
              r.v[8 + row % 8][col + 16 * (row / 8)] = goc::as_bits(float(int(random() % 33) - 16));
          uint32_t before[8][32];
          save(r, dst, before);
          ASSERT_EQ(floating[format](semantics, mask, 0, r.v + dst, r.v, r.v + 4, r.v + 8), 0);
          check(r, dst, mask, kFp8Dense[format], before);
        }
}

TEST(SubbyteWmma, AllFp8CodesThroughApi) {
  for (int format = 0; format < 4; ++format)
    for (int operand = 0; operand < 2; ++operand)
      for (int block = 0; block < 16; ++block) {
        Registers r;
        bool bf8 = operand ? format & 1 : format & 2;
        for (int index = 0; index < 16; ++index)
          for (int k = 0; k < 16; ++k) {
            // Each row/column places one code at K=0. The other operand
            // contains ones, including for nonfinite factors.
            set(r.v + 4 * operand, index, k, 8, k == 0 ? block * 16 + index : 0);
            bool other_bf8 = operand ? format & 2 : format & 1;
            set(r.v + 4 * (1 - operand), index, k, 8, other_bf8 ? 0x3c : 0x38);
          }
        ASSERT_EQ(floating[format](0, UINT64_MAX, 0, r.v + 16, r.v, r.v + 4, r.v + 8), 0);
        for (int row = 0; row < 16; ++row)
          for (int col = 0; col < 16; ++col) {
            int code = block * 16 + (operand ? col : row);
            int fraction_bits = bf8 ? 2 : 3, bias = bf8 ? 15 : 7;
            int exponent = (code & 127) >> fraction_bits,
                fraction = code & ((1 << fraction_bits) - 1);
            float value = goc::as_float(r.v[16 + row % 8][col + 16 * (row / 8)]);
            if ((bf8 && exponent == 31 && fraction) || (!bf8 && (code & 127) == 127)) {
              EXPECT_TRUE(std::isnan(value));
            } else {
              // Independent arithmetic decoding, including E4M3's finite top exponent.
              float want =
                  bf8 && exponent == 31
                      ? INFINITY
                      : std::ldexp(float(exponent ? (1 << fraction_bits) + fraction : fraction),
                                   (exponent ? exponent : 1) - bias - fraction_bits);
              if (code & 128)
                want = -want;
              EXPECT_EQ(value, want);
            }
          }
      }
}

TEST(SubbyteWmma, Fp8ModifiersAndStrictErrors) {
  for (auto fn : floating)
    for (uint32_t modifiers :
         {0u, GOC_WMMA_NEG_C, GOC_WMMA_ABS_C, GOC_WMMA_NEG_C | GOC_WMMA_ABS_C}) {
      Registers r;
      for (int reg = 0; reg < 8; ++reg)
        for (int lane = 0; lane < 32; ++lane)
          r.v[8 + reg][lane] = goc::as_bits(-3.0f);
      ASSERT_EQ(fn(GOC_FP16_OVFL, UINT64_MAX, modifiers, r.v + 16, r.v, r.v + 4, r.v + 8), 0);
      float want = (modifiers & GOC_WMMA_ABS_C) ? 3.0f : -3.0f;
      if (modifiers & GOC_WMMA_NEG_C)
        want = -want;
      for (int reg = 0; reg < 8; ++reg)
        for (int lane = 0; lane < 32; ++lane)
          EXPECT_EQ(r.v[16 + reg][lane], goc::as_bits(want));
      uint32_t before[8][32];
      save(r, 16, before);
      EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, UINT64_MAX, modifiers,
                   r.v + 16, r.v, r.v + 4, r.v + 8),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
      EXPECT_EQ(fn(0, UINT64_MAX, GOC_WMMA_NEG_LO_A, r.v + 16, r.v, r.v + 4, r.v + 8),
                GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(UINT64_C(1) << 63, UINT64_MAX, 0, r.v + 16, r.v, r.v + 4, r.v + 8),
                GOC_ERROR_INVALID_FLAGS);
      check(r, 16, 0, nullptr, before);
    }
}

TEST(SubbyteWmma, IntegerGoldensSignsClampMasksAndOverlap) {
  for (int shape = 0; shape < 3; ++shape)
    for (uint32_t mode = 0; mode < 8; ++mode)
      for (int dst : {0, 4, 8, 16})
        for (uint64_t mask : {UINT64_C(0), UINT64_C(0xdeadbeefa55a1234), UINT64_MAX})
          for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL}) {
            int bits = shape == 0 ? 8 : 4, K = shape == 2 ? 32 : 16;
            Registers r;
            std::minstd_rand random(12056925);
            for (int operand = 0; operand < 2; ++operand)
              for (int i = 0; i < 16 * K; ++i)
                set(r.v + 4 * operand, operand ? i % 16 : i / K, operand ? i / 16 : i % K, bits,
                    random() % (1u << bits));
            for (int row = 0; row < 16; ++row)
              for (int col = 0; col < 16; ++col) {
                int i = row * 16 + col;
                r.v[8 + row % 8][col + 16 * (row / 8)] = i % 4 == 0   ? 0x7ffffff0
                                                         : i % 4 == 1 ? 0x80000010
                                                                      : random();
              }
            uint32_t before[8][32];
            save(r, dst, before);
            uint32_t modifiers = (mode & 3) | ((mode & 4) ? GOC_WMMA_CLAMP : 0);
            ASSERT_EQ(integer[shape](semantics | GOC_SEMANTICS_STRICT | GOC_FP16_OVFL, mask,
                                     modifiers, r.v + dst, r.v, r.v + 4, r.v + 8),
                      0);
            check(r, dst, mask, kIntegerDense[shape][mode], before);
          }
}

TEST(SubbyteWmma, IntegerErrorsPreserveEveryDestination) {
  for (auto fn : integer) {
    Registers r;
    for (int reg = 0; reg < 8; ++reg)
      std::fill(r.v[16 + reg], r.v[16 + reg] + 32, 0x12345678);
    uint32_t before[8][32];
    save(r, 16, before);
    for (uint32_t flag :
         {GOC_WMMA_NEG_C, GOC_WMMA_NEG_HI_A, GOC_WMMA_NEG_HI_B, GOC_WMMA_ABS_C, UINT32_C(1) << 31})
      EXPECT_EQ(fn(0, UINT64_MAX, flag, r.v + 16, r.v, r.v + 4, r.v + 8), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(fn((UINT64_C(2) << 16) | GOC_SEMANTICS_STRICT, UINT64_MAX, 0, r.v + 16, r.v, r.v + 4,
                 r.v + 8),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    check(r, 16, 0, nullptr, before);
  }
}

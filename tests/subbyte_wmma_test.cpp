// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "goc/goc.h"
#include "integer_wmma_capture.h"
#include "integer_wmma_hardware.h"
#include "internal.h"
#include "subbyte_golden.h"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <initializer_list>
#include <random>
#include <stdint.h>
#include <vector>

namespace {

using Wmma = decltype(&goc_v_wmma_f32_16x16x16_fp8_fp8);

const Wmma floating[] = {goc_v_wmma_f32_16x16x16_fp8_fp8, goc_v_wmma_f32_16x16x16_fp8_bf8,
                         goc_v_wmma_f32_16x16x16_bf8_fp8, goc_v_wmma_f32_16x16x16_bf8_bf8};
const Wmma integer[] = {goc_v_wmma_i32_16x16x16_iu8_rdna4, goc_v_wmma_i32_16x16x16_iu4_rdna4,
                        goc_v_wmma_i32_16x16x32_iu4};

std::vector<uint64_t> cpu_levels() {
  std::vector<uint64_t> levels;
  for (uint64_t level = GOC_CPU_BASELINE; level <= goc_init_cpu_flags(); ++level)
    levels.push_back(level);
  return levels;
}

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

void check(Registers &r, int dst, const uint32_t *golden, const uint32_t (&before)[8][32]) {
  for (int row = 0; row < 16; ++row)
    for (int col = 0; col < 16; ++col) {
      int lane = col + 16 * (row / 8), reg = row % 8;
      EXPECT_EQ(r.v[dst + reg][lane], golden ? golden[row * 16 + col] : before[reg][lane])
          << "row=" << row << " col=" << col;
    }
  r.guards();
}

void save(Registers &r, int dst, uint32_t (&before)[8][32]) {
  for (int reg = 0; reg < 8; ++reg)
    std::copy(r.v[dst + reg], r.v[dst + reg] + 32, before[reg]);
}

} // namespace

TEST(SubbyteWmma, Fp8DenseGoldensAndOverlap) {
  for (uint64_t cpu : cpu_levels())
    for (uint32_t mode : {0U, GOC_WMMA_NEG_C, GOC_WMMA_ABS_C, GOC_WMMA_NEG_C | GOC_WMMA_ABS_C})
      for (int format = 0; format < 4; ++format)
        for (int dst : {0, 4, 8, 16})

          for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL}) {
            Registers r;
            std::minstd_rand random(12056925);
            for (int operand = 0; operand < 2; ++operand) {
              bool bf8 = operand ? format & 1 : format & 2;
              for (int i = 0; i < 256; ++i) {
                uint32_t x = random();
                uint32_t code = (x & 128) | ((bf8 ? 48 : 32) + x % (bf8 ? 24 : 48));
                set(r.v + 4 * operand, operand ? i % 16 : i / 16, operand ? i / 16 : i % 16, 8,
                    code);
              }
            }
            uint32_t expected[256];
            for (int row = 0; row < 16; ++row)
              for (int col = 0; col < 16; ++col) {
                int c = int(random() % 33) - 16;
                r.v[8 + row % 8][col + 16 * (row / 8)] = goc::as_bits(float(c));
                int adjusted = mode & GOC_WMMA_ABS_C ? std::abs(c) : c;
                if (mode & GOC_WMMA_NEG_C)
                  adjusted = -adjusted;
                expected[row * 16 + col] = goc::as_bits(
                    goc::as_float(kFp8Dense[format][row * 16 + col]) + float(adjusted - c));
              }
            uint32_t before[8][32];
            save(r, dst, before);
            ASSERT_EQ(floating[format](cpu | semantics, mode, r.v + dst, r.v, r.v + 4, r.v + 8), 0);
            check(r, dst, expected, before);
          }
}

TEST(SubbyteWmma, AllFp8CodesThroughApi) {
  for (uint64_t cpu : cpu_levels())
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
          ASSERT_EQ(floating[format](cpu, 0, r.v + 16, r.v, r.v + 4, r.v + 8), 0);
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
      ASSERT_EQ(fn(GOC_FP16_OVFL, modifiers, r.v + 16, r.v, r.v + 4, r.v + 8), 0);
      float want = (modifiers & GOC_WMMA_ABS_C) ? 3.0f : -3.0f;
      if (modifiers & GOC_WMMA_NEG_C)
        want = -want;
      for (int reg = 0; reg < 8; ++reg)
        for (int lane = 0; lane < 32; ++lane)
          EXPECT_EQ(r.v[16 + reg][lane], goc::as_bits(want));
      uint32_t before[8][32];
      save(r, 16, before);
      EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, modifiers, r.v + 16, r.v,
                   r.v + 4, r.v + 8),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
      EXPECT_EQ(fn(0, GOC_WMMA_NEG_LO_A, r.v + 16, r.v, r.v + 4, r.v + 8), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(1ULL << 63, 0, r.v + 16, r.v, r.v + 4, r.v + 8), GOC_ERROR_INVALID_FLAGS);
      check(r, 16, nullptr, before);
    }
}

TEST(SubbyteWmma, IntegerGoldensSignsClampAndOverlap) {
  for (uint64_t cpu : cpu_levels())
    for (int shape = 0; shape < 3; ++shape)
      for (uint32_t mode = 0; mode < 8; ++mode)
        for (int dst : {0, 4, 8, 16})

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
            ASSERT_EQ(integer[shape](cpu | semantics | GOC_SEMANTICS_STRICT | GOC_FP16_OVFL,
                                     modifiers, r.v + dst, r.v, r.v + 4, r.v + 8),
                      0);
            check(r, dst, kIntegerDense[shape][mode], before);
          }
}

TEST(SubbyteWmma, IntegerErrorsPreserveEveryDestination) {
  for (uint64_t cpu : cpu_levels())
    for (auto fn : integer) {
      Registers r;
      for (int reg = 0; reg < 8; ++reg)
        std::fill(r.v[16 + reg], r.v[16 + reg] + 32, 0x12345678);
      uint32_t before[8][32];
      save(r, 16, before);
      for (uint32_t flag :
           {GOC_WMMA_NEG_C, GOC_WMMA_NEG_HI_A, GOC_WMMA_NEG_HI_B, GOC_WMMA_ABS_C, 1U << 31})
        EXPECT_EQ(fn(cpu, flag, r.v + 16, r.v, r.v + 4, r.v + 8), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(cpu | (2ULL << 16) | GOC_SEMANTICS_STRICT, 0, r.v + 16, r.v, r.v + 4, r.v + 8),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
      check(r, 16, nullptr, before);
    }
}

// Compute the reference from logical matrices, independently of the SIMD
// unpacking, paired products, and overflow detection.
TEST(SubbyteWmma, IntegerExtremesAndCancellationAcrossCpuLevels) {
  for (int shape = 0; shape < 3; ++shape)
    for (int mode = 0; mode < 8; ++mode)
      for (int pattern = 0; pattern < 7; ++pattern) {
        const int bits = shape == 0 ? 8 : 4, k_size = shape == 2 ? 32 : 16;
        const int range = 1 << bits;
        uint32_t codes[2][16][32] = {};
        int64_t values[2][16][32] = {};
        std::minstd_rand random(991 + pattern);
        for (int operand = 0; operand < 2; ++operand)
          for (int index = 0; index < 16; ++index)
            for (int k = 0; k < k_size; ++k) {
              int code = 0;
              switch (pattern) {
              case 0:
                code = range - 1;
                break; // unsigned-byte pair sums exceed int16
              case 1:
                code = range / 2;
                break; // most-negative signed factors
              case 2:
                code = range / 2 - 1;
                break;
              case 3: // Products cancel after potentially overflowing in either direction.
              case 6:
                code = operand == 0
                           ? 1
                           : ((k < k_size / 2) != (pattern == 6) ? range / 2 - 1 : range / 2);
                break;
              case 4:
                code = (index * k_size + k) % range;
                break;
              default:
                code = random() % range;
                break;
              }
              codes[operand][index][k] = uint32_t(code);
              values[operand][index][k] = code;
              if ((mode & (1 << operand)) && code >= range / 2)
                values[operand][index][k] -= range;
            }
        const int64_t accumulators[] = {INT32_MAX, INT32_MIN, INT32_MAX - 100, INT32_MIN + 100,
                                        0,         -1,        123456,          -123456};
        uint32_t golden[256];
        for (int row = 0; row < 16; ++row)
          for (int col = 0; col < 16; ++col) {
            int64_t acc = accumulators[(row + col) % 8];
            for (int half = 0; half < 2; ++half) {
              for (int k = 0; k < k_size; ++k)
                if ((k / 8) % 2 == half)
                  acc += values[0][row][k] * values[1][col][k];
              if (mode & 4)
                acc = std::clamp(acc, int64_t(INT32_MIN), int64_t(INT32_MAX));
            }
            golden[row * 16 + col] = uint32_t(acc);
          }
        for (uint64_t cpu : cpu_levels()) {
          SCOPED_TRACE(::testing::Message() << "shape=" << shape << " mode=" << mode
                                            << " pattern=" << pattern << " cpu=" << cpu);
          Registers r;
          for (int operand = 0; operand < 2; ++operand)
            for (int index = 0; index < 16; ++index)
              for (int k = 0; k < k_size; ++k)
                set(r.v + 4 * operand, index, k, bits, codes[operand][index][k]);
          for (int row = 0; row < 16; ++row)
            for (int col = 0; col < 16; ++col)
              r.v[8 + row % 8][col + 16 * (row / 8)] = uint32_t(accumulators[(row + col) % 8]);
          uint32_t before[8][32];
          save(r, 16, before);
          const uint32_t modifiers = (mode & 3) | ((mode & 4) ? GOC_WMMA_CLAMP : 0);
          ASSERT_EQ(integer[shape](cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                                   modifiers, r.v + 16, r.v, r.v + 4, r.v + 8),
                    0);
          check(r, 16, golden, before);
        }
      }
}

TEST(SubbyteWmma, IntegerHardwareStagedClampCorpus) {
  for (unsigned shape = 0; shape < 3; ++shape)
    for (uint64_t cpu : cpu_levels())
      for (unsigned sample = 0; sample < 16; ++sample)
        for (unsigned mode = 0; mode < 8; ++mode) {
          uint32_t data[15][32], result[8][32];
          goc_test::integer_wmma_capture_inputs(sample, data);
          const uint32_t *a[] = {data[0], data[1]}, *b[] = {data[2], data[3]}, *c[8];
          uint32_t *d[8];
          for (unsigned reg = 0; reg < 8; ++reg) {
            c[reg] = data[6 + reg];
            d[reg] = result[reg];
          }
          uint32_t modifiers = (mode & 3) | ((mode & 4) ? GOC_WMMA_CLAMP : 0);
          ASSERT_EQ(integer[shape](cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                                   modifiers, d, a, b, c),
                    GOC_SUCCESS);
          uint64_t digest = goc_test::capture_hash_seed;
          for (const auto &reg : result)
            for (uint32_t word : reg)
              digest = goc_test::capture_hash_bytes(digest, word, 4);
          EXPECT_EQ(digest, goc_test::dense_integer_capture_digests[shape][sample][mode])
              << shape << "/" << cpu << "/" << sample << "/" << mode;
        }
}

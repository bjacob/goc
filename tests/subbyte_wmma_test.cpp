// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "fp_environment.h"
#include "fuzzy.h"
#include "goc/goc.h"
#include "integer_wmma_capture.h"
#include "integer_wmma_hardware.h"
#include "integer_wmma_wave64_hardware.h"
#include "loose_tolerances.h"
#include "matrix_capture_state.h"
#include "rdna_wmma_f32_bf8_bf8_cases.h"
#include "rdna_wmma_f32_bf8_fp8_cases.h"
#include "rdna_wmma_f32_fp8_bf8_cases.h"
#include "rdna_wmma_f32_fp8_fp8_cases.h"
#include "subbyte_golden.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <random>
#include <stdint.h>
#include <vector>

namespace {

uint32_t as_bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float as_float(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

// Registers have reversed pointer order and one leading guard word.
// Every word outside the destination, including out-of-wave storage, stays raw.
template <size_t Words>
void preserved(const uint32_t (&actual)[24][Words], const uint32_t (&before)[24][Words], int dst,
               int width) {
  for (int reg = 0; reg < 24; ++reg)
    for (size_t word = 0; word < Words; ++word)
      if (reg < dst || reg >= dst + 256 / width || word == 0 || word > size_t(width)) {
        EXPECT_EQ(actual[23 - reg][word], before[23 - reg][word])
            << "reg=" << reg << " word=" << word;
      }
}

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
                r.v[8 + row % 8][col + 16 * (row / 8)] = as_bits(float(c));
                expected[row * 16 + col] = kFp8Dense[format][row * 16 + col];
              }
            uint32_t before[8][32];
            save(r, dst, before);
            uint32_t original[24][35];
            std::memcpy(original, r.storage, sizeof(original));
            ASSERT_EQ(floating[format](cpu | semantics, mode, r.v + dst, r.v, r.v + 4, r.v + 8), 0);
            check(r, dst, expected, before);
            preserved(r.storage, original, dst, 32);
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
              float value = as_float(r.v[16 + row % 8][col + 16 * (row / 8)]);
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
          r.v[8 + reg][lane] = as_bits(-3.0f);
      ASSERT_EQ(fn(GOC_FP16_OVFL, modifiers, r.v + 16, r.v, r.v + 4, r.v + 8), 0);
      float want = -3.0f;
      for (int reg = 0; reg < 8; ++reg)
        for (int lane = 0; lane < 32; ++lane)
          EXPECT_EQ(r.v[16 + reg][lane], as_bits(want));
      uint32_t before[8][32];
      save(r, 16, before);
      uint32_t original[24][35];
      std::memcpy(original, r.storage, sizeof(original));
      EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, modifiers, r.v + 16, r.v,
                   r.v + 4, r.v + 8),
                GOC_SUCCESS);
      for (unsigned bit = 0; bit < 64; ++bit)
        if (!((1ULL << bit) & (GOC_WMMA_NEG_C | GOC_WMMA_ABS_C))) {
          EXPECT_EQ(fn(0, 1ULL << bit, r.v + 16, r.v, r.v + 4, r.v + 8), GOC_ERROR_INVALID_FLAGS);
          EXPECT_EQ(std::memcmp(original, r.storage, sizeof(original)), 0);
        }
      EXPECT_EQ(fn(1ULL << 63, 0, r.v + 16, r.v, r.v + 4, r.v + 8), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(std::memcmp(original, r.storage, sizeof(original)), 0);
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
            uint32_t original[24][35];
            std::memcpy(original, r.storage, sizeof(original));
            uint32_t modifiers = (mode & 3) | ((mode & 4) ? GOC_WMMA_CLAMP : 0);
            ASSERT_EQ(integer[shape](cpu | semantics | GOC_SEMANTICS_STRICT | GOC_FP16_OVFL,
                                     modifiers, r.v + dst, r.v, r.v + 4, r.v + 8),
                      0);
            check(r, dst, kIntegerDense[shape][mode], before);
            preserved(r.storage, original, dst, 32);
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
          uint32_t original[15][32];
          std::memcpy(original, data, sizeof(original));
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
          EXPECT_EQ(std::memcmp(original, data, sizeof(original)), 0);
        }
}

namespace {

const Wmma floating_wave64[] = {
    goc_v_wmma_f32_16x16x16_fp8_fp8_wave64, goc_v_wmma_f32_16x16x16_fp8_bf8_wave64,
    goc_v_wmma_f32_16x16x16_bf8_fp8_wave64, goc_v_wmma_f32_16x16x16_bf8_bf8_wave64};

struct WideRegisters {
  uint32_t storage[24][66] = {};
  uint32_t *v[24];

  WideRegisters() {
    for (int i = 0; i < 24; ++i) {
      storage[i][0] = storage[i][65] = 0xdeadbeef;
      v[i] = storage[23 - i] + 1;
    }
  }
};

// Encode the manual's bit-field layout, independently of the implementation's
// pointer views: source lane = {k[2], k[3], index[3:0]}.
void set_wave64(uint32_t *const *v, int index, int k, int bits, uint32_t value) {
  ASSERT_EQ(bits, 8);
  int lane = index | ((k & 4) << 3) | ((k & 8) << 1);
  int shift = 8 * (k & 3);
  v[0][lane] = (v[0][lane] & ~(255U << shift)) | (value << shift);
}

void save_wave64(WideRegisters &r, int dst, uint32_t (&before)[4][64]) {
  for (int reg = 0; reg < 4; ++reg)
    std::copy_n(r.v[dst + reg], 64, before[reg]);
}

void check_wave64(WideRegisters &r, int dst, const uint32_t *golden,
                  const uint32_t (&before)[4][64]) {
  for (int row = 0; row < 16; ++row)
    for (int col = 0; col < 16; ++col) {
      int lane = col | ((row & 4) << 3) | ((row & 8) << 1);
      int reg = row & 3;
      EXPECT_EQ(r.v[dst + reg][lane], golden ? golden[row * 16 + col] : before[reg][lane]);
    }
  for (const auto &reg : r.storage) {
    EXPECT_EQ(reg[0], 0xdeadbeefU);
    EXPECT_EQ(reg[65], 0xdeadbeefU);
  }
}

} // namespace

TEST(SubbyteWmma, Wave64Fp8DenseGoldensAndOverlap) {
  for (uint64_t cpu : cpu_levels())
    for (uint32_t mode : {0U, GOC_WMMA_NEG_C, GOC_WMMA_ABS_C, GOC_WMMA_NEG_C | GOC_WMMA_ABS_C})
      for (int format = 0; format < 4; ++format)
        for (int dst : {0, 4, 8, 16})
          for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL}) {
            WideRegisters r;
            std::minstd_rand random(12056925);
            for (int operand = 0; operand < 2; ++operand) {
              bool bf8 = operand ? format & 1 : format & 2;
              for (int i = 0; i < 256; ++i) {
                uint32_t x = random();
                uint32_t code = (x & 128) | ((bf8 ? 48 : 32) + x % (bf8 ? 24 : 48));
                set_wave64(r.v + 4 * operand, operand ? i % 16 : i / 16, operand ? i / 16 : i % 16,
                           8, code);
              }
            }
            uint32_t expected[256];
            for (int row = 0; row < 16; ++row)
              for (int col = 0; col < 16; ++col) {
                int c = int(random() % 33) - 16;
                r.v[8 + row % 4][col + 16 * (row / 8) + 32 * ((row / 4) % 2)] = as_bits(float(c));
                expected[row * 16 + col] = kFp8Dense[format][row * 16 + col];
              }
            uint32_t before[4][64];
            save_wave64(r, dst, before);
            uint32_t original[24][66];
            std::memcpy(original, r.storage, sizeof(original));
            ASSERT_EQ(
                floating_wave64[format](cpu | semantics, mode, r.v + dst, r.v, r.v + 4, r.v + 8),
                0);
            check_wave64(r, dst, expected, before);
            preserved(r.storage, original, dst, 64);
          }
}

TEST(SubbyteWmma, Wave64AllFp8CodesThroughApi) {
  for (uint64_t cpu : cpu_levels())
    for (int format = 0; format < 4; ++format)
      for (int operand = 0; operand < 2; ++operand)
        for (int block = 0; block < 16; ++block) {
          WideRegisters r;
          bool bf8 = operand ? format & 1 : format & 2;
          for (int index = 0; index < 16; ++index)
            for (int k = 0; k < 16; ++k) {
              // Each row/column places one code at K=0. The other operand
              // contains ones, including for nonfinite factors.
              set_wave64(r.v + 4 * operand, index, k, 8, k == 0 ? block * 16 + index : 0);
              bool other_bf8 = operand ? format & 2 : format & 1;
              set_wave64(r.v + 4 * (1 - operand), index, k, 8, other_bf8 ? 0x3c : 0x38);
            }
          ASSERT_EQ(floating_wave64[format](cpu, 0, r.v + 16, r.v, r.v + 4, r.v + 8), 0);
          for (int row = 0; row < 16; ++row)
            for (int col = 0; col < 16; ++col) {
              int code = block * 16 + (operand ? col : row);
              int fraction_bits = bf8 ? 2 : 3, bias = bf8 ? 15 : 7;
              int exponent = (code & 127) >> fraction_bits,
                  fraction = code & ((1 << fraction_bits) - 1);
              float value =
                  as_float(r.v[16 + row % 4][col + 16 * (row / 8) + 32 * ((row / 4) % 2)]);
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

TEST(SubbyteWmma, Wave64Fp8ModifiersAndStrictErrors) {
  for (auto fn : floating_wave64)
    for (uint32_t modifiers :
         {0u, GOC_WMMA_NEG_C, GOC_WMMA_ABS_C, GOC_WMMA_NEG_C | GOC_WMMA_ABS_C}) {
      WideRegisters r;
      for (int reg = 0; reg < 4; ++reg)
        for (int lane = 0; lane < 64; ++lane)
          r.v[8 + reg][lane] = as_bits(-3.0f);
      ASSERT_EQ(fn(GOC_FP16_OVFL, modifiers, r.v + 16, r.v, r.v + 4, r.v + 8), 0);
      float want = -3.0f;
      for (int reg = 0; reg < 4; ++reg)
        for (int lane = 0; lane < 64; ++lane)
          EXPECT_EQ(r.v[16 + reg][lane], as_bits(want));
      uint32_t before[4][64];
      save_wave64(r, 16, before);
      uint32_t original[24][66];
      std::memcpy(original, r.storage, sizeof(original));
      EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, modifiers, r.v + 16, r.v,
                   r.v + 4, r.v + 8),
                GOC_SUCCESS);
      for (unsigned bit = 0; bit < 64; ++bit)
        if (!((1ULL << bit) & (GOC_WMMA_NEG_C | GOC_WMMA_ABS_C))) {
          EXPECT_EQ(fn(0, 1ULL << bit, r.v + 16, r.v, r.v + 4, r.v + 8), GOC_ERROR_INVALID_FLAGS);
          EXPECT_EQ(std::memcmp(original, r.storage, sizeof(original)), 0);
        }
      EXPECT_EQ(fn(1ULL << 63, 0, r.v + 16, r.v, r.v + 4, r.v + 8), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(std::memcmp(original, r.storage, sizeof(original)), 0);
      check_wave64(r, 16, nullptr, before);
    }
}

// RX 9070 gfx1201, native Wave64; capture_fp8_wmma_wave64.cpp.
// Finite, exactly representable sums validate the complete register mapping.
// They do not distinguish a common permutation of A/B's K dimension or
// establish bit-exact floating accumulation for general inputs.
TEST(SubbyteWmma, Wave64Fp8HardwareCorpus) {
  const uint32_t codes[2][8] = {{0x38, 0x40, 0x48, 0x50, 0xb8, 0xc0, 0xc8, 0xd0},
                                {0x3c, 0x40, 0x44, 0x48, 0xbc, 0xc0, 0xc4, 0xc8}};
  for (uint64_t cpu : cpu_levels())
    for (unsigned format = 0; format < 4; ++format)
      for (int dst : {0, 4, 8, 16}) {
        WideRegisters r;
        for (unsigned lane = 0; lane < 64; ++lane) {
          for (unsigned byte = 0; byte < 4; ++byte) {
            unsigned x = (lane * 3 + lane / 7 + byte * 5) % 8;
            unsigned y = (lane * 5 + lane / 3 + byte * 3) % 8;
            r.v[0][lane] |= codes[bool(format & 2)][x] << (byte * 8);
            r.v[4][lane] |= codes[bool(format & 1)][y] << (byte * 8);
          }
          for (unsigned reg = 0; reg < 4; ++reg)
            r.v[8 + reg][lane] = as_bits(float(int(lane) - 32 + 3 * int(reg)));
        }
        ASSERT_EQ(floating_wave64[format](cpu, 0, r.v + dst, r.v, r.v + 4, r.v + 8), GOC_SUCCESS);
        uint64_t hash = goc_test::capture_hash_seed;
        for (unsigned reg = 0; reg < 4; ++reg)
          for (unsigned lane = 0; lane < 64; ++lane)
            hash = goc_test::capture_hash_word(hash, r.v[dst + reg][lane]);
        EXPECT_EQ(hash, 0xcebccc43bff33725ULL) << cpu << "/" << format << "/" << dst;
      }
}

namespace {

const Wmma integer_wave64[] = {goc_v_wmma_i32_16x16x16_iu8_rdna4_wave64,
                               goc_v_wmma_i32_16x16x16_iu4_rdna4_wave64,
                               goc_v_wmma_i32_16x16x32_iu4_wave64};

} // namespace

TEST(SubbyteWmma, Wave64IntegerNativeStagedClampCorpusAndAliases) {
  for (unsigned shape = 0; shape < 3; ++shape)
    for (uint64_t cpu : cpu_levels())
      for (unsigned sample = 0; sample < 16; ++sample)
        for (unsigned mode = 0; mode < 8; ++mode)
          for (unsigned dst : {0U, 2U, 6U, 10U}) {
            uint32_t data[15][64];
            goc_test::integer_wmma_capture_inputs(sample, data);
            uint32_t original[15][64];
            std::memcpy(original, data, sizeof(original));
            const uint32_t *a[] = {data[0]}, *b[] = {data[2]}, *c[4];
            uint32_t *d[4];
            for (unsigned reg = 0; reg < 4; ++reg) {
              c[reg] = data[6 + reg];
              d[reg] = data[dst + reg];
            }
            uint32_t modifiers = (mode & 3) | ((mode & 4) ? GOC_WMMA_CLAMP : 0);
            ASSERT_EQ(
                integer_wave64[shape](cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                                      modifiers, d, a, b, c),
                GOC_SUCCESS);
            uint64_t hash = goc_test::capture_hash_seed;
            for (unsigned reg = 0; reg < 4; ++reg)
              for (unsigned lane = 0; lane < 64; ++lane)
                hash = goc_test::capture_hash_word(hash, d[reg][lane]);
            EXPECT_EQ(hash, goc_test::integer_wmma_wave64_digests[shape][sample][mode])
                << shape << "/" << cpu << "/" << sample << "/" << mode << "/" << dst;
            for (unsigned reg = 0; reg < 15; ++reg)
              if (reg < dst || reg >= dst + 4) {
                EXPECT_EQ(std::memcmp(original[reg], data[reg], sizeof(original[reg])), 0);
              }
          }
}

TEST(SubbyteWmma, Wave64IntegerValidationPreservesOutputs) {
  for (auto fn : integer_wave64) {
    WideRegisters r;
    uint32_t before[4][64];
    save_wave64(r, 16, before);
    for (uint64_t invalid : {uint64_t(4), uint64_t(1ULL << 32), uint64_t(1ULL << 63)})
      EXPECT_EQ(fn(0, invalid, r.v + 16, r.v, r.v + 4, r.v + 8), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(fn(1ULL << 63, 0, r.v + 16, r.v, r.v + 4, r.v + 8), GOC_ERROR_INVALID_FLAGS);
    check_wave64(r, 16, nullptr, before);
  }
}

namespace {

// Direct encoding of RDNA4 table 42's Wave64 input bit fields.
void set_integer_wave64(uint32_t *const *v, int index, int k, int bits, int k_size,
                        uint32_t value) {
  int lane = index;
  if (bits == 8)
    lane |= ((k & 4) << 3) | ((k & 8) << 1);
  else if (k_size == 32)
    lane |= ((k & 8) << 2) | (k & 16);
  else
    lane |= (k & 8) << 1;
  int shift = (k % (32 / bits)) * bits;
  uint32_t mask = ((1U << bits) - 1) << shift;
  v[0][lane] = (v[0][lane] & ~mask) | (value << shift);
}

} // namespace

TEST(SubbyteWmma, Wave64IntegerExtremesAndCancellationAcrossCpuLevels) {
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
                if (k / (k_size / 2) == half)
                  acc += values[0][row][k] * values[1][col][k];
              if (mode & 4)
                acc = std::clamp(acc, int64_t(INT32_MIN), int64_t(INT32_MAX));
            }
            golden[row * 16 + col] = uint32_t(acc);
          }
        for (uint64_t cpu : cpu_levels()) {
          SCOPED_TRACE(::testing::Message() << "shape=" << shape << " mode=" << mode
                                            << " pattern=" << pattern << " cpu=" << cpu);
          WideRegisters r;
          for (int operand = 0; operand < 2; ++operand)
            for (int index = 0; index < 16; ++index)
              for (int k = 0; k < k_size; ++k)
                set_integer_wave64(r.v + 4 * operand, index, k, bits, k_size,
                                   codes[operand][index][k]);
          for (int row = 0; row < 16; ++row)
            for (int col = 0; col < 16; ++col)
              r.v[8 + row % 4][col + 16 * (row / 8) + 32 * ((row / 4) % 2)] =
                  uint32_t(accumulators[(row + col) % 8]);
          uint32_t before[4][64];
          save_wave64(r, 16, before);
          const uint32_t modifiers = (mode & 3) | ((mode & 4) ? GOC_WMMA_CLAMP : 0);
          ASSERT_EQ(
              integer_wave64[shape](cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                                    modifiers, r.v + 16, r.v, r.v + 4, r.v + 8),
              0);
          check_wave64(r, 16, golden, before);
        }
      }
}

TEST(SubbyteWmma, FractionalFp8MatricesAgainstDoubleReference) {
  for (int width : {32, 64})
    for (unsigned format = 0; format < 4; ++format) {
      std::mt19937 random(52813);
      for (unsigned batch = 0; batch < 64; ++batch) {
        WideRegisters input;
        double values[2][16][16], expected[16][16];
        for (unsigned operand = 0; operand < 2; ++operand) {
          bool bf8 = operand ? format & 1 : format & 2;
          unsigned mantissa = bf8 ? 2 : 3, bias = bf8 ? 15 : 7;
          for (int index = 0; index < 16; ++index)
            for (int k = 0; k < 16; ++k) {
              unsigned bits = random();
              unsigned exponent = 1 + ((bits >> 8) % (bias + 1));
              unsigned fraction = bits & ((1U << mantissa) - 1);
              unsigned code = (bits >> 24 & 128) | (exponent << mantissa) | fraction;
              double value = std::ldexp(double((1U << mantissa) + fraction),
                                        int(exponent) - int(bias) - int(mantissa));
              values[operand][index][k] = code & 128 ? -value : value;
              if (width == 64)
                set_wave64(input.v + 4 * operand, index, k, 8, code);
              else
                set(input.v + 4 * operand, index, k, 8, code);
            }
        }
        unsigned modifiers = (batch & 1 ? GOC_WMMA_ABS_C : 0) | (batch & 2 ? GOC_WMMA_NEG_C : 0);
        for (int row = 0; row < 16; ++row)
          for (int col = 0; col < 16; ++col) {
            float c = float(int(random() % 2000001) - 1000000) / 262144.0f;
            int lane = col + 16 * (row / 8) + (width == 64 ? 32 * (row / 4 % 2) : 0);
            input.v[8 + row % (256 / width)][lane] = as_bits(c);
            double sum = c;
            for (int k = 0; k < 16; ++k)
              sum += values[0][row][k] * values[1][col][k];
            expected[row][col] = sum;
          }
        // Cycle the four destination layouts independently of C modifiers.
        const int destinations[] = {0, 4, 8, 16};
        int dst = destinations[(batch / 4) % 4];
        for (uint64_t cpu : cpu_levels()) {
          WideRegisters r;
          std::memcpy(r.storage, input.storage, sizeof(r.storage));
          ASSERT_EQ((width == 64 ? floating_wave64[format] : floating[format])(
                        cpu, modifiers, r.v + dst, r.v, r.v + 4, r.v + 8),
                    GOC_SUCCESS);
          for (int row = 0; row < 16; ++row)
            for (int col = 0; col < 16; ++col) {
              int lane = col + 16 * (row / 8) + (width == 64 ? 32 * (row / 4 % 2) : 0);
              float actual = as_float(r.v[dst + row % (256 / width)][lane]);
              EXPECT_TRUE(goc_test::fuzzy(actual, expected[row][col], goc_test::wmma_f32_tolerance))
                  << width << '/' << format << '/' << batch << '/' << cpu << " row=" << row
                  << " col=" << col;
            }
          preserved(r.storage, input.storage, dst, width);
        }
      }
    }
}

TEST(SubbyteWmma, NativeFp8CapturedMatricesAndOperands) {
  goc_test::ScopedFpEnvironment restore;
  ASSERT_TRUE(restore.saved());
  for (unsigned host_case = 0; host_case < 8; ++host_case) {
    ASSERT_NO_FATAL_FAILURE(goc_test::matrix_test_host(host_case));
    const unsigned controls = goc_test::matrix_host_controls();

    struct Shape {
      bool abf8, bbf8;
      decltype(&goc_v_wmma_f32_16x16x16_fp8_fp8) wave32, wave64;
      const RdnaWmmaFloatMatrixCase *matrices;
      unsigned matrix_count;
      const RdnaWmmaFloatOperandCase *operands;
      unsigned operand_count;
    };

    const Shape shapes[] = {
        {false, false, goc_v_wmma_f32_16x16x16_fp8_fp8, goc_v_wmma_f32_16x16x16_fp8_fp8_wave64,
         kRdnaWmmaF32Fp8Fp8Matrices.data(), kRdnaWmmaF32Fp8Fp8Matrices.size(),
         kRdnaWmmaF32Fp8Fp8Operands.data(), kRdnaWmmaF32Fp8Fp8Operands.size()},
        {false, true, goc_v_wmma_f32_16x16x16_fp8_bf8, goc_v_wmma_f32_16x16x16_fp8_bf8_wave64,
         kRdnaWmmaF32Fp8Bf8Matrices.data(), kRdnaWmmaF32Fp8Bf8Matrices.size(),
         kRdnaWmmaF32Fp8Bf8Operands.data(), kRdnaWmmaF32Fp8Bf8Operands.size()},
        {true, false, goc_v_wmma_f32_16x16x16_bf8_fp8, goc_v_wmma_f32_16x16x16_bf8_fp8_wave64,
         kRdnaWmmaF32Bf8Fp8Matrices.data(), kRdnaWmmaF32Bf8Fp8Matrices.size(),
         kRdnaWmmaF32Bf8Fp8Operands.data(), kRdnaWmmaF32Bf8Fp8Operands.size()},
        {true, true, goc_v_wmma_f32_16x16x16_bf8_bf8, goc_v_wmma_f32_16x16x16_bf8_bf8_wave64,
         kRdnaWmmaF32Bf8Bf8Matrices.data(), kRdnaWmmaF32Bf8Bf8Matrices.size(),
         kRdnaWmmaF32Bf8Bf8Operands.data(), kRdnaWmmaF32Bf8Bf8Operands.size()},
    };
    for (const auto &shape : shapes)
      for (unsigned sample = 0; sample < shape.matrix_count + shape.operand_count; ++sample)
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
          const bool matrix = sample < shape.matrix_count;
          const auto &m = shape.matrices[matrix ? sample : 0];
          const auto &o = shape.operands[matrix ? 0 : sample - shape.matrix_count];
          const unsigned width = matrix ? m.width : o.width;
          const auto code = matrix ? m.code : o.code;
          SCOPED_TRACE(::testing::Message()
                       << shape.abf8 << shape.bbf8 << "/" << sample << "/" << width << "/" << cpu);
          uint32_t data[33][66], expected[33][66];
          uint32_t *v[33];
          uint32_t rng = m.seed;
          constexpr uint32_t accumulators[] = {0,           1,           0x80000000U, 0x3f800001U,
                                               0xbf800001U, 0x43800001U, 0xc3800001U, 0x80000001U};
          constexpr uint32_t pairs[] = {4, 8, 12, 9, 13, 14};
          for (unsigned reg = 0; reg < 33; ++reg) {
            v[reg] = data[reg] + 1;
            std::fill(data[reg], data[reg] + 66, 0xdeadbeefU);
            for (unsigned lane = 0; lane < width; ++lane) {
              uint32_t value;
              if (matrix) {
                rng ^= rng << 13;
                rng ^= rng >> 17;
                rng ^= rng << 5;
                value = rng;
                if (m.finite && reg < 16) {
                  const unsigned max_finite = (reg < 8 ? shape.abf8 : shape.bbf8) ? 0x7b : 0x7e;
                  value = 0;
                  for (unsigned shift = 0; shift < 32; shift += 8) {
                    unsigned part = (rng >> shift) & 255;
                    value |= ((part & 128) | ((part & 127) % (max_finite + 1))) << shift;
                  }
                } else if (m.finite && reg < 32)
                  value = accumulators[(rng >> 24) & 7];
                if (reg == 32) {
                  value = 0;
                  for (unsigned i = 0; i < 8; ++i)
                    value |= pairs[((rng >> (4 * i)) & 15) % 6] << (4 * i);
                }
              } else
                value = reg < 8 ? o.a : reg < 16 ? o.b : reg < 24 ? o.c : reg < 32 ? o.d : o.index;
              v[reg][lane] = value;
            }
          }
          if (!matrix && o.factor_base != RdnaWmmaFloatOperandCase::kUniformFactors)
            for (unsigned i = 0; i < 128; ++i) {
              unsigned offset = o.factor_base * width + i;
              v[offset / width][offset % width] = o.factor_groups[i / 16];
            }
          std::memcpy(expected, data, sizeof(expected));
          unsigned dest = code[0] & 255;
          unsigned count = matrix ? m.expected_words : 256;
          for (unsigned i = 0; i < count; ++i)
            expected[dest + i / width][1 + i % width] = matrix ? m.expected[i] : o.expected;
          // These saved dense encodings select VGPR sources. C ABS/NEG are
          // accepted aliases even though the hardware observations ignore them.
          unsigned a = (code[1] & 511) - 256, b = ((code[1] >> 9) & 511) - 256,
                   c = (code[1] >> 18) & 511;
          ASSERT_LT(a, 33U);
          ASSERT_LT(b, 33U);
          uint32_t inline_c[8][64];
          const uint32_t *cv[8];
          if (c >= 256) {
            ASSERT_LT(c - 256, 33U);
            for (unsigned i = 0; i < 256 / width; ++i)
              cv[i] = v[c - 256 + i];
          } else {
            // VOP inline 0, integer -1, and FP32 0.5 in this capture corpus.
            ASSERT_TRUE(c == 128 || c == 193 || c == 240);
            uint32_t bits = c == 128 ? 0 : c == 193 ? 0xffffffffU : 0x3f000000U;
            for (unsigned i = 0; i < 8; ++i) {
              std::fill(inline_c[i], inline_c[i] + 64, bits);
              cv[i] = inline_c[i];
            }
          }
          uint64_t modifiers = (code[0] & (1U << 10) ? GOC_WMMA_ABS_C : 0) |
                               (code[1] & (1U << 31) ? GOC_WMMA_NEG_C : 0);
          ASSERT_EQ((width == 32 ? shape.wave32 : shape.wave64)(
                        cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT |
                            goc_test::matrix_capture_flags(matrix ? m.mode : o.mode, host_case),
                        modifiers, v + dest, v + a, v + b, cv),
                    GOC_SUCCESS);
          for (unsigned reg = 0; reg < 33; ++reg)
            for (unsigned lane = 0; lane < 66; ++lane)
              ASSERT_EQ(data[reg][lane], expected[reg][lane])
                  << "reg=" << reg << " guarded lane=" << lane;
        }
    EXPECT_EQ(goc_test::matrix_host_controls(), controls);
  }
}

// Literal witnesses from ad2bb973ec4's dense operand captures. The two Wave64
// DOT8 steps read sixteen-lane groups 0+2, then 1+3. Opposing infinities within
// a step override C's NaN, while infinities in separate steps preserve C's NaN.
TEST(SubbyteWmma, NativeDot8InfinityPrecedenceAndWave64Order) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (bool same_step : {false, true}) {
      uint32_t a[64], b[64], c[4][64], d[4][64];
      const uint32_t *av[] = {a}, *bv[] = {b}, *cv[] = {c[0], c[1], c[2], c[3]};
      uint32_t *dv[] = {d[0], d[1], d[2], d[3]};
      const uint32_t groups[] = {0x7c7c7c7cU, same_step ? 0x3c3c3c3cU : 0xfcfcfcfcU,
                                 same_step ? 0xfcfcfcfcU : 0x7c7c7c7cU,
                                 same_step ? 0x3c3c3c3cU : 0xfcfcfcfcU};
      for (unsigned lane = 0; lane < 64; ++lane) {
        a[lane] = groups[lane / 16];
        b[lane] = 0x3c3c3c3cU;
        for (auto &reg : c)
          reg[lane] = 0xffc54321U;
      }
      ASSERT_EQ(goc_v_wmma_f32_16x16x16_bf8_bf8_wave64(
                    cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, dv, av, bv, cv),
                GOC_SUCCESS);
      for (const auto &reg : d)
        for (uint32_t word : reg)
          EXPECT_EQ(word, same_step ? 0xffc00a3dU : 0xffc54321U);
    }
}

TEST(SubbyteWmma, NativeFp8SubnormalFactorAndIgnoredCFlags) {
  // Includes native E4M3FN byte 0x01, with captured C=-0.5. All four C flag
  // combinations have the same captured answer, 0x44e0f040.
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (uint64_t flags : {uint64_t(0), uint64_t(GOC_WMMA_ABS_C), uint64_t(GOC_WMMA_NEG_C),
                           uint64_t(GOC_WMMA_ABS_C | GOC_WMMA_NEG_C)}) {
      uint32_t a[2][32], b[2][32], c[8][32], d[8][32];
      const uint32_t *av[] = {a[0], a[1]}, *bv[] = {b[0], b[1]}, *cv[8];
      uint32_t *dv[8];
      for (unsigned reg = 0; reg < 8; ++reg) {
        cv[reg] = c[reg];
        dv[reg] = d[reg];
        std::fill(c[reg], c[reg] + 32, 0xbf000000U);
      }
      for (unsigned reg = 0; reg < 2; ++reg) {
        std::fill(a[reg], a[reg] + 32, 0x013840b8U);
        std::fill(b[reg], b[reg] + 32, 0x407e3801U);
      }
      ASSERT_EQ(goc_v_wmma_f32_16x16x16_fp8_fp8(cpu | GOC_SEMANTICS_EXACT_EMPIRICAL |
                                                    GOC_SEMANTICS_STRICT,
                                                flags, dv, av, bv, cv),
                GOC_SUCCESS);
      for (const auto &reg : d)
        for (uint32_t word : reg)
          EXPECT_EQ(word, 0x44e0f040U);
    }
}

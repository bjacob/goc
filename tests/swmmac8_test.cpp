// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "fuzzy.h"
#include "goc/goc.h"
#include "loose_tolerances.h"
#include "swmmac8_hardware.h"
#include "swmmac8_wave64_hardware.h"

#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <limits>
#include <random>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_v_swmmac_f32_16x16x32_fp8_fp8);
const Fn functions[] = {goc_v_swmmac_f32_16x16x32_fp8_fp8, goc_v_swmmac_f32_16x16x32_fp8_bf8,
                        goc_v_swmmac_f32_16x16x32_bf8_fp8, goc_v_swmmac_f32_16x16x32_bf8_bf8};

uint32_t mode(unsigned variant) { return variant ? GOC_SWMMAC_INDEX_KEY_1 : 0; }

template <unsigned Lanes> struct LaneRegisters {
  static const unsigned a_count = Lanes == 64 ? 1 : 2;
  static const unsigned b_count = Lanes == 64 ? 2 : 4;
  static const unsigned d_count = Lanes == 64 ? 4 : 8;
  uint32_t data[15][Lanes];
  const uint32_t *a[a_count], *b[b_count], *index[1];
  uint32_t *d[d_count];

  explicit LaneRegisters(unsigned op, unsigned sample = 0) {
    goc_test::swmmac8_capture_inputs(op, sample, data);
    for (unsigned reg = 0; reg < a_count; ++reg)
      a[reg] = data[reg];
    for (unsigned reg = 0; reg < b_count; ++reg)
      b[reg] = data[2 + reg];
    for (unsigned reg = 0; reg < d_count; ++reg)
      d[reg] = data[6 + reg];
    index[0] = data[14];
  }
};

using Registers = LaneRegisters<32>;
using WideRegisters = LaneRegisters<64>;

} // namespace

TEST(Swmmac8, HardwareAllModifiers) {
  // A non-strict exact request must follow the same loose path.
  for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
    for (unsigned op = 0; op < 4; ++op)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (unsigned variant = 0; variant < 2; ++variant)
          for (unsigned sample = 0; sample < 16; ++sample) {
            Registers r(op, sample);
            uint32_t before[15][32];
            std::memcpy(before, r.data, sizeof(before));
            ASSERT_EQ(functions[op](cpu | semantics, mode(variant), r.d, r.a, r.b, r.index),
                      GOC_SUCCESS);
            uint64_t digest = goc_test::capture_hash_seed;
            for (unsigned reg = 0; reg < 8u; ++reg)
              for (unsigned lane = 0; lane < 32; ++lane)
                digest = goc_test::capture_hash_bytes(digest, r.d[reg][lane], 4);
            ASSERT_EQ(digest, goc_test::swmmac8_capture_digests[op][sample][variant])
                << op << "/" << semantics << "/" << cpu << "/" << variant << "/" << sample;
            for (unsigned reg = 0; reg < 15; ++reg)
              if (reg < 6 || reg >= 14) {
                EXPECT_EQ(std::memcmp(before[reg], r.data[reg], sizeof(before[reg])), 0);
              }
          }
}

TEST(Swmmac8, EveryLaneAndModifier) {
  // Differential CPU diagnostic; the captures above are the independent oracle.
  for (unsigned op = 0; op < 4; ++op)
    for (unsigned variant = 0; variant < 2; ++variant) {
      Registers full(op);
      ASSERT_EQ(functions[op](0, mode(variant), full.d, full.a, full.b, full.index), GOC_SUCCESS);
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
        Registers r(op);
        ASSERT_EQ(functions[op](cpu, mode(variant), r.d, r.a, r.b, r.index), GOC_SUCCESS);
        for (unsigned reg = 0; reg < 8u; ++reg)
          for (unsigned lane = 0; lane < 32; ++lane)
            ASSERT_EQ(r.d[reg][lane], full.d[reg][lane]) << op << "/" << cpu << "/" << variant;
      }
    }
}

TEST(Swmmac8, OverlappingSourcesAndDuplicateDestinations) {
  for (unsigned op = 0; op < 4; ++op)
    for (unsigned variant = 0; variant < 2; ++variant)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (unsigned alias = 0; alias < 4; ++alias) {
          Registers r(op), reference(op);
          unsigned regs = 8;
          unsigned offsets[8];
          for (unsigned reg = 0; reg < regs; ++reg) {
            offsets[reg] = alias == 0 ? reg % 2 : alias == 1 ? 2 + reg % 4 : alias == 2 ? 14 : 6;
            r.d[reg] = r.data[offsets[reg]];
            std::memcpy(reference.d[reg], r.d[reg], 32 * sizeof(uint32_t));
          }

          ASSERT_EQ(functions[op](0, mode(variant), reference.d, reference.a, reference.b,
                                  reference.index),
                    GOC_SUCCESS);
          uint32_t expected[15][32];
          std::memcpy(expected, r.data, sizeof(expected));
          for (unsigned reg = 0; reg < regs; ++reg)
            for (unsigned lane = 0; lane < 32; ++lane)
              expected[offsets[reg]][lane] = reference.d[reg][lane];
          ASSERT_EQ(functions[op](cpu, mode(variant), r.d, r.a, r.b, r.index), GOC_SUCCESS);
          ASSERT_EQ(std::memcmp(expected, r.data, sizeof(expected)), 0)
              << op << "/" << cpu << "/" << variant << "/" << alias;
        }
}

TEST(Swmmac8, Validation) {
  for (Fn fn : functions) {
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, nullptr, nullptr, nullptr,
                 nullptr),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    Registers r(0);
    uint32_t before[15][32];
    std::memcpy(before, r.data, sizeof(before));
    for (unsigned bit = 0; bit < 64; ++bit)
      if (bit != 7) {
        EXPECT_EQ(fn(0, 1ULL << bit, r.d, r.a, r.b, r.index), GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(std::memcmp(before, r.data, sizeof(before)), 0);
      }
    EXPECT_EQ(fn(1ULL << 63, 0, r.d, r.a, r.b, r.index), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, r.d, r.a, r.b, r.index),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    EXPECT_EQ(std::memcmp(before, r.data, sizeof(before)), 0);
  }
}

TEST(Swmmac8, EveryInputEncoding) {
  for (unsigned op = 0; op < 4; ++op)
    for (unsigned source = 0; source < 2; ++source)
      for (unsigned code = 0; code < 256; ++code) {
        bool bf8 = source ? op % 2 : op / 2;
        unsigned mantissa_bits = bf8 ? 2 : 3;
        unsigned exponent = (code & 127) >> mantissa_bits;
        unsigned fraction = code & ((1u << mantissa_bits) - 1);
        float value;
        if (bf8 && exponent == 31)
          value = fraction ? std::numeric_limits<float>::quiet_NaN()
                           : std::numeric_limits<float>::infinity();
        else if (!bf8 && exponent == 15 && fraction == 7)
          value = std::numeric_limits<float>::quiet_NaN();
        else
          value = std::ldexp(float(fraction + (exponent ? 1u << mantissa_bits : 0)),
                             int(exponent ? exponent : 1) - (bf8 ? 15 : 7) - int(mantissa_bits));
        if (code & 128)
          value = -value;
        float want = std::fma(16.0f, value, 0.0f);
        uint32_t want_bits;
        std::memcpy(&want_bits, &want, sizeof(want));
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
          Registers r(op);
          unsigned av = source == 0 ? code : op / 2 ? 0x3c : 0x38;
          unsigned bv = source == 1 ? code : op % 2 ? 0x3c : 0x38;
          for (unsigned reg = 0; reg < 6; ++reg)
            for (auto &word : r.data[reg])
              word = (reg < 2 ? av : bv) * 0x01010101u;
          for (unsigned reg = 6; reg < 14; ++reg)
            for (auto &word : r.data[reg])
              word = 0;
          ASSERT_EQ(functions[op](cpu, 0, r.d, r.a, r.b, r.index), GOC_SUCCESS);
          for (unsigned reg = 0; reg < 8; ++reg)
            for (unsigned lane = 0; lane < 32; ++lane) {
              if (std::isnan(want)) {
                EXPECT_GT(r.d[reg][lane] & 0x7fffffffu, 0x7f800000u);
              } else {
                EXPECT_EQ(r.d[reg][lane], want_bits) << op << "/" << source << "/" << code;
              }
            }
        }
      }
}

namespace {

const Fn wide_functions[] = {
    goc_v_swmmac_f32_16x16x32_fp8_fp8_wave64, goc_v_swmmac_f32_16x16x32_fp8_bf8_wave64,
    goc_v_swmmac_f32_16x16x32_bf8_fp8_wave64, goc_v_swmmac_f32_16x16x32_bf8_bf8_wave64};

} // namespace

TEST(Swmmac8, Wave64NativeAllFormatsAndIndexKeys) {
  // A non-strict exact request must follow the same loose path.
  for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
    for (unsigned op = 0; op < 4; ++op)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (unsigned key = 0; key < 4; ++key)
          for (unsigned sample = 0; sample < 16; ++sample) {
            uint32_t data[15][64];
            goc_test::swmmac8_capture_inputs(op, sample, data);
            uint32_t before[15][64];
            std::memcpy(before, data, sizeof(before));
            const uint32_t *a[] = {data[0]}, *b[] = {data[2], data[3]}, *index[] = {data[14]};
            uint32_t *d[] = {data[6], data[7], data[8], data[9]};
            ASSERT_EQ(wide_functions[op](cpu | semantics, key << 7, d, a, b, index), GOC_SUCCESS);
            uint64_t hash = goc_test::capture_hash_seed;
            for (unsigned reg = 0; reg < 4; ++reg)
              for (unsigned lane = 0; lane < 64; ++lane)
                hash = goc_test::capture_hash_word(hash, d[reg][lane]);
            ASSERT_EQ(hash, goc_test::swmmac8_wave64_digests[op][sample][key])
                << op << "/" << semantics << "/" << cpu << "/" << sample << "/" << key;
            for (unsigned reg = 0; reg < 15; ++reg)
              if (reg < 6 || reg >= 10) {
                EXPECT_EQ(std::memcmp(before[reg], data[reg], sizeof(before[reg])), 0);
              }
          }
}

TEST(Swmmac8, Wave64OverlappingSourcesAndDuplicateDestinations) {
  for (unsigned op = 0; op < 4; ++op)
    for (unsigned variant = 0; variant < 4; ++variant)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (unsigned alias = 0; alias < 4; ++alias) {
          WideRegisters r(op), reference(op);
          unsigned regs = 4;
          unsigned offsets[4];
          for (unsigned reg = 0; reg < regs; ++reg) {
            offsets[reg] = alias == 0 ? 0 : alias == 1 ? 2 + reg % 2 : alias == 2 ? 14 : 6;
            r.d[reg] = r.data[offsets[reg]];
            std::memcpy(reference.d[reg], r.d[reg], 64 * sizeof(uint32_t));
          }

          ASSERT_EQ(wide_functions[op](0, variant << 7, reference.d, reference.a, reference.b,
                                       reference.index),
                    GOC_SUCCESS);
          uint32_t expected[15][64];
          std::memcpy(expected, r.data, sizeof(expected));
          for (unsigned reg = 0; reg < regs; ++reg)
            for (unsigned lane = 0; lane < 64; ++lane)
              expected[offsets[reg]][lane] = reference.d[reg][lane];
          ASSERT_EQ(wide_functions[op](cpu, variant << 7, r.d, r.a, r.b, r.index), GOC_SUCCESS);
          ASSERT_EQ(std::memcmp(expected, r.data, sizeof(expected)), 0)
              << op << "/" << cpu << "/" << variant << "/" << alias;
        }
}

TEST(Swmmac8, Wave64EveryInputEncoding) {
  for (unsigned op = 0; op < 4; ++op)
    for (unsigned source = 0; source < 2; ++source)
      for (unsigned code = 0; code < 256; ++code) {
        bool bf8 = source ? op % 2 : op / 2;
        unsigned mantissa_bits = bf8 ? 2 : 3;
        unsigned exponent = (code & 127) >> mantissa_bits;
        unsigned fraction = code & ((1u << mantissa_bits) - 1);
        float value;
        if (bf8 && exponent == 31)
          value = fraction ? std::numeric_limits<float>::quiet_NaN()
                           : std::numeric_limits<float>::infinity();
        else if (!bf8 && exponent == 15 && fraction == 7)
          value = std::numeric_limits<float>::quiet_NaN();
        else
          value = std::ldexp(float(fraction + (exponent ? 1u << mantissa_bits : 0)),
                             int(exponent ? exponent : 1) - (bf8 ? 15 : 7) - int(mantissa_bits));
        if (code & 128)
          value = -value;
        float want = std::fma(16.0f, value, 0.0f);
        uint32_t want_bits;
        std::memcpy(&want_bits, &want, sizeof(want));
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
          WideRegisters r(op);
          unsigned av = source == 0 ? code : op / 2 ? 0x3c : 0x38;
          unsigned bv = source == 1 ? code : op % 2 ? 0x3c : 0x38;
          for (unsigned reg = 0; reg < 6; ++reg)
            for (auto &word : r.data[reg])
              word = (reg < 2 ? av : bv) * 0x01010101u;
          for (unsigned reg = 6; reg < 14; ++reg)
            for (auto &word : r.data[reg])
              word = 0;
          ASSERT_EQ(wide_functions[op](cpu, 0, r.d, r.a, r.b, r.index), GOC_SUCCESS);
          for (unsigned reg = 0; reg < 4; ++reg)
            for (unsigned lane = 0; lane < 64; ++lane) {
              if (std::isnan(want)) {
                EXPECT_GT(r.d[reg][lane] & 0x7fffffffu, 0x7f800000u);
              } else {
                EXPECT_EQ(r.d[reg][lane], want_bits) << op << "/" << source << "/" << code;
              }
            }
        }
      }
}

TEST(Swmmac8, Wave64ValidationPreservesOutputs) {
  for (Fn fn : wide_functions) {
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, nullptr, nullptr, nullptr,
                 nullptr),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    WideRegisters r(0);
    uint32_t before[15][64];
    std::memcpy(before, r.data, sizeof before);
    for (unsigned bit = 0; bit < 64; ++bit)
      if (bit != 7 && bit != 8) {
        EXPECT_EQ(fn(0, 1ULL << bit, r.d, r.a, r.b, r.index), GOC_ERROR_INVALID_FLAGS);
      }
    EXPECT_EQ(fn(1ULL << 63, 0, r.d, r.a, r.b, r.index), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, r.d, r.a, r.b, r.index),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    EXPECT_EQ(std::memcmp(before, r.data, sizeof before), 0);
  }
}

TEST(Swmmac8, FractionalMatricesAgainstDoubleReference) {
  // Construct logical compressed A, dense B and legal 2:4 selectors, then pack
  // their ISA layouts. No production decoders or accumulation helpers are used.
  // Values stay within [-4,4]; the reference is mathematical, not bit-exact.
  for (unsigned width : {32U, 64U})
    for (unsigned op = 0; op < 4; ++op) {
      std::mt19937 random(71849);
      auto operand = [&](bool bf8, uint8_t &code) {
        unsigned bits = random();
        unsigned mantissa = bf8 ? 2 : 3, bias = bf8 ? 15 : 7;
        unsigned exponent = 1 + ((bits >> 8) % (bias + 1));
        unsigned fraction = bits & ((1U << mantissa) - 1);
        code = uint8_t((bits >> 24 & 128) | (exponent << mantissa) | fraction);
        double value = std::ldexp(double((1U << mantissa) + fraction),
                                  int(exponent) - int(bias) - int(mantissa));
        return code & 128 ? -value : value;
      };
      for (unsigned batch = 0; batch < 64; ++batch)
        for (unsigned key = 0; key < width / 16; ++key) {
          uint32_t original[15][64] = {};
          double a_values[16][16], b_values[32][16], sums[16][16];
          unsigned selected[16][16];
          for (unsigned row = 0; row < 16; ++row)
            for (unsigned pair = 0; pair < 8; ++pair) {
              const unsigned pairs[] = {4, 8, 12, 9, 13, 14};
              unsigned selection = pairs[random() % 6];
              for (unsigned part = 0; part < 2; ++part) {
                unsigned ck = 2 * pair + part;
                unsigned slot = ck % (width == 64 ? 4 : 8);
                unsigned lane = row + 16 * (ck / 8) + (width == 64 ? 32 * (ck / 4 % 2) : 0);
                selected[row][ck] = 4 * pair + ((selection >> (2 * part)) & 3);
                original[14][lane] |= ((selection >> (2 * part)) & 3)
                                      << (key * (width == 64 ? 8 : 16) + 2 * slot);
                uint8_t code;
                a_values[row][ck] = operand(op / 2, code);
                original[slot / 4][lane] |= uint32_t(code) << (8 * (slot % 4));
              }
            }
          for (unsigned k = 0; k < 32; ++k)
            for (unsigned col = 0; col < 16; ++col) {
              uint8_t code;
              b_values[k][col] = operand(op % 2, code);
              unsigned lane = col + 16 * (k / 16) + (width == 64 ? 32 * (k / 8 % 2) : 0);
              unsigned reg = (k % (width == 64 ? 8 : 16)) / 4;
              original[2 + reg][lane] |= uint32_t(code) << (8 * (k % 4));
            }
          for (unsigned row = 0; row < 16; ++row)
            for (unsigned col = 0; col < 16; ++col) {
              float c = float(int(random() % 2000001) - 1000000) / 262144.0f;
              unsigned reg = row % (width == 64 ? 4 : 8);
              unsigned lane = col + 16 * (row / 8) + (width == 64 ? 32 * (row / 4 % 2) : 0);
              std::memcpy(&original[6 + reg][lane], &c, sizeof(c));
              double sum = c;
              for (unsigned ck = 0; ck < 16; ++ck)
                sum += a_values[row][ck] * b_values[selected[row][ck]][col];
              sums[row][col] = sum;
            }
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
            uint32_t data[15][64];
            std::memcpy(data, original, sizeof(data));
            const uint32_t *a[] = {data[0], data[1]};
            const uint32_t *b[] = {data[2], data[3], data[4], data[5]}, *index[] = {data[14]};
            uint32_t *d[] = {data[6],  data[7],  data[8],  data[9],
                             data[10], data[11], data[12], data[13]};
            ASSERT_EQ(
                (width == 64 ? wide_functions[op] : functions[op])(cpu, key << 7, d, a, b, index),
                GOC_SUCCESS);
            for (unsigned row = 0; row < 16; ++row)
              for (unsigned col = 0; col < 16; ++col) {
                unsigned reg = row % (width == 64 ? 4 : 8);
                unsigned lane = col + 16 * (row / 8) + (width == 64 ? 32 * (row / 4 % 2) : 0);
                float actual;
                std::memcpy(&actual, &d[reg][lane], sizeof(actual));
                EXPECT_TRUE(goc_test::fuzzy(actual, sums[row][col], goc_test::wmma_f32_tolerance))
                    << width << '/' << op << '/' << batch << '/' << key << '/' << cpu
                    << " row=" << row << " col=" << col;
              }
            for (unsigned reg = 0; reg < 15; ++reg)
              for (unsigned lane = 0; lane < 64; ++lane)
                if (reg < 6 || reg >= 6 + 256U / width || lane >= width) {
                  EXPECT_EQ(data[reg][lane], original[reg][lane]);
                }
          }
        }
    }
}

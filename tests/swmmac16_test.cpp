// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "fuzzy.h"
#include "goc/goc.h"
#include "loose_tolerances.h"
#include "rdna_swmmac_bf16_bf16_cases.h"
#include "rdna_swmmac_f16_f16_cases.h"
#include "rdna_swmmac_f32_bf16_cases.h"
#include "rdna_swmmac_f32_f16_cases.h"
#include "swmmac16_hardware.h"
#include "swmmac16_wave64_hardware.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdint.h>
#include <string>

namespace {

using Fn = decltype(&goc_v_swmmac_f32_16x16x32_f16);
const Fn functions[] = {goc_v_swmmac_f32_16x16x32_f16, goc_v_swmmac_f32_16x16x32_bf16,
                        goc_v_swmmac_f16_16x16x32_f16, goc_v_swmmac_bf16_16x16x32_bf16};

uint32_t mode(unsigned variant) {
  return (variant & 3) | ((variant & 12) << 1) | ((variant >> 4) << 7);
}

template <unsigned Lanes> struct LaneRegisters {
  static const unsigned a_count = Lanes == 64 ? 2 : 4;
  static const unsigned b_count = Lanes == 64 ? 4 : 8;
  uint32_t data[21][Lanes];
  const uint32_t *a[a_count], *b[b_count], *index[1];
  uint32_t *d[b_count];

  explicit LaneRegisters(unsigned op) {
    goc_test::swmmac16_capture_inputs(op, data);
    for (unsigned r = 0; r < a_count; ++r)
      a[r] = data[r];
    for (unsigned r = 0; r < b_count; ++r) {
      b[r] = data[4 + r];
      d[r] = data[12 + r];
    }
    index[0] = data[20];
  }
};

using Registers = LaneRegisters<32>;
using WideRegisters = LaneRegisters<64>;

} // namespace

TEST(Swmmac16, HardwareAllModifiers) {
  // Exactly representable small-integer captures agree in both semantics.
  for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
    for (unsigned op = 0; op < 4; ++op)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (unsigned variant = 0; variant < 32; ++variant) {
          Registers r(op);
          uint32_t before[21][32];
          std::memcpy(before, r.data, sizeof(before));
          ASSERT_EQ(functions[op](cpu | semantics, mode(variant), r.d, r.a, r.b, r.index),
                    GOC_SUCCESS);
          uint64_t digest = goc_test::capture_hash_seed;
          for (unsigned reg = 0; reg < (op >= 2 ? 4u : 8u); ++reg)
            for (unsigned lane = 0; lane < 32; ++lane)
              digest = goc_test::capture_hash_bytes(digest, r.d[reg][lane], 4);
          ASSERT_EQ(digest, goc_test::swmmac16_capture_digests[op][variant])
              << op << "/" << semantics << "/" << cpu << "/" << variant;
          for (unsigned reg = 0; reg < 21; ++reg)
            if (reg < 12 || reg >= 12 + (op >= 2 ? 4U : 8U)) {
              EXPECT_EQ(std::memcmp(before[reg], r.data[reg], sizeof(before[reg])), 0);
            }
        }
}

TEST(Swmmac16, EveryLaneAndModifier) {
  // Differential CPU diagnostic; the captures above are the independent oracle.
  for (unsigned op = 0; op < 4; ++op)
    for (unsigned variant = 0; variant < 32; ++variant) {
      Registers full(op);
      ASSERT_EQ(functions[op](0, mode(variant), full.d, full.a, full.b, full.index), GOC_SUCCESS);
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
        Registers r(op);
        ASSERT_EQ(functions[op](cpu, mode(variant), r.d, r.a, r.b, r.index), GOC_SUCCESS);
        for (unsigned reg = 0; reg < (op >= 2 ? 4u : 8u); ++reg)
          for (unsigned lane = 0; lane < 32; ++lane)
            ASSERT_EQ(r.d[reg][lane], full.d[reg][lane]) << op << "/" << cpu << "/" << variant;
      }
    }
}

TEST(Swmmac16, OverlappingSourcesAndDuplicateDestinations) {
  for (unsigned op = 0; op < 4; ++op)
    for (unsigned variant = 0; variant < 32; ++variant)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (unsigned alias = 0; alias < 4; ++alias) {
          Registers r(op), reference(op);
          unsigned regs = op >= 2 ? 4 : 8;
          unsigned offsets[8];
          for (unsigned reg = 0; reg < regs; ++reg) {
            offsets[reg] = alias == 0 ? reg % 4 : alias == 1 ? 4 + reg : alias == 2 ? 20 : 12;
            r.d[reg] = r.data[offsets[reg]];
            std::memcpy(reference.d[reg], r.d[reg], 32 * sizeof(uint32_t));
          }

          ASSERT_EQ(functions[op](0, mode(variant), reference.d, reference.a, reference.b,
                                  reference.index),
                    GOC_SUCCESS);
          uint32_t expected[21][32];
          std::memcpy(expected, r.data, sizeof(expected));
          for (unsigned reg = 0; reg < regs; ++reg)
            for (unsigned lane = 0; lane < 32; ++lane)
              expected[offsets[reg]][lane] = reference.d[reg][lane];
          ASSERT_EQ(functions[op](cpu, mode(variant), r.d, r.a, r.b, r.index), GOC_SUCCESS);
          ASSERT_EQ(std::memcmp(expected, r.data, sizeof(expected)), 0)
              << op << "/" << cpu << "/" << variant << "/" << alias;
        }
}

TEST(Swmmac16, Validation) {
  for (Fn fn : functions) {
    EXPECT_EQ(fn((2ULL << 16) | GOC_SEMANTICS_STRICT, 0, nullptr, nullptr, nullptr, nullptr),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    Registers r(0);
    uint32_t before[21][32];
    std::memcpy(before, r.data, sizeof(before));
    for (unsigned bit = 0; bit < 64; ++bit)
      if (!((1ULL << bit) & mode(31))) {
        EXPECT_EQ(fn(0, 1ULL << bit, r.d, r.a, r.b, r.index), GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(std::memcmp(before, r.data, sizeof(before)), 0);
      }
    EXPECT_EQ(fn(1ULL << 63, 0, r.d, r.a, r.b, r.index), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(fn((2ULL << 16) | GOC_SEMANTICS_STRICT, 0, r.d, r.a, r.b, r.index),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    EXPECT_EQ(std::memcmp(before, r.data, sizeof(before)), 0);
  }
}

TEST(Swmmac16, PackedIntermediateNarrowing) {
  for (unsigned op : {2u, 3u})
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned odd = 0; odd < 2; ++odd) {
        Registers r(op);
        // Each DOT4 adds only one eighth of an output ULP at 1.0.
        // Narrowing each step discards it even when C has an odd significand.
        uint32_t a = op == 2 ? 0x0200 : 0x3980;
        uint32_t one = op == 2 ? 0x3c00 : 0x3f80;
        for (unsigned reg = 0; reg < 4; ++reg)
          for (auto &word : r.data[reg])
            word = a | (a << 16);
        for (unsigned reg = 4; reg < 12; ++reg)
          for (auto &word : r.data[reg])
            word = one | (one << 16);
        uint32_t c = one + odd;
        for (unsigned reg = 12; reg < 16; ++reg)
          for (auto &word : r.data[reg])
            word = c | (c << 16);
        ASSERT_EQ(functions[op](cpu, 0, r.d, r.a, r.b, r.index), GOC_SUCCESS);
        uint32_t want = one + odd;
        for (unsigned reg = 0; reg < 4; ++reg)
          for (unsigned lane = 0; lane < 32; ++lane)
            EXPECT_EQ(r.d[reg][lane], want | (want << 16));
      }
}

TEST(Swmmac16, FiniteOverflowAndInfinity) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (bool saturate : {false, true})
      for (bool infinity : {false, true})
        for (bool negative : {false, true}) {
          Registers r(2);
          uint32_t av = infinity ? 0x7c00 : 0x5c00; // Infinity or 256.
          uint32_t bv = infinity ? 0x3c00 : 0x5c00;
          for (unsigned reg = 0; reg < 4; ++reg)
            for (auto &word : r.data[reg])
              word = av | (av << 16);
          for (unsigned reg = 4; reg < 12; ++reg)
            for (auto &word : r.data[reg])
              word = bv | (bv << 16);
          for (unsigned reg = 12; reg < 16; ++reg)
            for (auto &word : r.data[reg])
              word = 0;
          uint32_t modifiers = negative ? GOC_WMMA_NEG_LO_A | GOC_WMMA_NEG_HI_A : 0;
          ASSERT_EQ(
              functions[2](cpu | (saturate ? GOC_FP16_OVFL : 0), modifiers, r.d, r.a, r.b, r.index),
              GOC_SUCCESS);
          uint32_t want = (saturate && !infinity ? 0x7bff : 0x7c00) | (negative ? 0x8000 : 0);
          for (unsigned reg = 0; reg < 4; ++reg)
            for (unsigned lane = 0; lane < 32; ++lane)
              EXPECT_EQ(r.d[reg][lane], want | (want << 16));
        }
}

TEST(Swmmac16, AccumulatorAndPairNegationAcrossOutputFormats) {
  for (unsigned op = 0; op < 4; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < 3; ++variant) {
        Registers r(op);
        uint32_t one = op & 1 ? 0x3f803f80 : 0x3c003c00;
        for (unsigned reg = 0; reg < 12; ++reg)
          for (unsigned lane = 0; lane < 32; ++lane)
            r.data[reg][lane] = one;
        for (unsigned reg = 12; reg < 20; ++reg)
          for (unsigned lane = 0; lane < 32; ++lane)
            r.data[reg][lane] = op >= 2 ? 0x40004000 : 0x40000000; // Initial accumulator: 2.
        for (unsigned lane = 0; lane < 32; ++lane)
          r.data[20][lane] = 0x44444444; // Select increasing positions 0 and 1.
        uint32_t flags = variant == 0   ? 0
                         : variant == 1 ? GOC_WMMA_NEG_LO_B
                                        : GOC_WMMA_NEG_LO_B | GOC_WMMA_NEG_HI_B;
        ASSERT_EQ(functions[op](cpu, flags, r.d, r.a, r.b, r.index), GOC_SUCCESS);
        // Sixteen products of one: 2+16, 2+8-8, or 2-16.
        const uint32_t fp32[] = {0x41900000, 0x40000000, 0xc1600000};
        const uint32_t fp16[] = {0x4c80, 0x4000, 0xcb00};
        uint32_t expected = op < 2    ? fp32[variant]
                            : op == 2 ? fp16[variant] * 0x10001u
                                      : (fp32[variant] >> 16) * 0x10001u;
        for (unsigned reg = 0; reg < (op >= 2 ? 4u : 8u); ++reg)
          for (unsigned lane = 0; lane < 32; ++lane)
            ASSERT_EQ(r.d[reg][lane], expected) << op << '/' << cpu << '/' << variant;
      }
}

namespace {

const Fn wide_functions[] = {
    goc_v_swmmac_f32_16x16x32_f16_wave64, goc_v_swmmac_f32_16x16x32_bf16_wave64,
    goc_v_swmmac_f16_16x16x32_f16_wave64, goc_v_swmmac_bf16_16x16x32_bf16_wave64};

} // namespace

TEST(Swmmac16, Wave64NativeAllModifiers) {
  // Exactly representable small-integer captures agree in both semantics.
  for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
    for (unsigned op = 0; op < 4; ++op)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (unsigned variant = 0; variant < 64; ++variant) {
          uint32_t data[21][64];
          goc_test::swmmac16_capture_inputs(op, data);
          uint32_t before[21][64];
          std::memcpy(before, data, sizeof(before));
          const uint32_t *a[] = {data[0], data[1]}, *b[] = {data[4], data[5], data[6], data[7]},
                         *index[] = {data[20]};
          uint32_t *d[] = {data[12], data[13], data[14], data[15]};
          ASSERT_EQ(wide_functions[op](cpu | semantics, mode(variant), d, a, b, index),
                    GOC_SUCCESS);
          uint64_t hash = goc_test::capture_hash_seed;
          for (unsigned reg = 0; reg < (op >= 2 ? 2U : 4U); ++reg)
            for (unsigned lane = 0; lane < 64; ++lane)
              hash = goc_test::capture_hash_word(hash, d[reg][lane]);
          ASSERT_EQ(hash, goc_test::swmmac16_wave64_digests[op][variant])
              << op << "/" << semantics << "/" << cpu << "/" << variant;
          for (unsigned reg = 0; reg < 21; ++reg)
            if (reg < 12 || reg >= 12 + (op >= 2 ? 2U : 4U)) {
              EXPECT_EQ(std::memcmp(before[reg], data[reg], sizeof(before[reg])), 0);
            }
        }
}

TEST(Swmmac16, Wave64OverlappingSourcesAndDuplicateDestinations) {
  for (unsigned op = 0; op < 4; ++op)
    for (unsigned variant = 0; variant < 64; ++variant)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (unsigned alias = 0; alias < 4; ++alias) {
          WideRegisters r(op), reference(op);
          unsigned regs = op >= 2 ? 2 : 4;
          unsigned offsets[4];
          for (unsigned reg = 0; reg < regs; ++reg) {
            offsets[reg] = alias == 0 ? reg % 2 : alias == 1 ? 4 + reg : alias == 2 ? 20 : 12;
            r.d[reg] = r.data[offsets[reg]];
            std::memcpy(reference.d[reg], r.d[reg], 64 * sizeof(uint32_t));
          }

          ASSERT_EQ(wide_functions[op](0, mode(variant), reference.d, reference.a, reference.b,
                                       reference.index),
                    GOC_SUCCESS);
          uint32_t expected[21][64];
          std::memcpy(expected, r.data, sizeof(expected));
          for (unsigned reg = 0; reg < regs; ++reg)
            for (unsigned lane = 0; lane < 64; ++lane)
              expected[offsets[reg]][lane] = reference.d[reg][lane];
          ASSERT_EQ(wide_functions[op](cpu, mode(variant), r.d, r.a, r.b, r.index), GOC_SUCCESS);
          ASSERT_EQ(std::memcmp(expected, r.data, sizeof(expected)), 0)
              << op << "/" << cpu << "/" << variant << "/" << alias;
        }
}

TEST(Swmmac16, Wave64Validation) {
  for (Fn fn : wide_functions) {
    EXPECT_EQ(fn((2ULL << 16) | GOC_SEMANTICS_STRICT, 0, nullptr, nullptr, nullptr, nullptr),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    WideRegisters r(0);
    uint32_t before[21][64];
    std::memcpy(before, r.data, sizeof(before));
    for (unsigned bit = 0; bit < 64; ++bit)
      if (!((1ULL << bit) & mode(63))) {
        EXPECT_EQ(fn(0, 1ULL << bit, r.d, r.a, r.b, r.index), GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(std::memcmp(before, r.data, sizeof(before)), 0);
      }
    EXPECT_EQ(fn(1ULL << 63, 0, r.d, r.a, r.b, r.index), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(fn((2ULL << 16) | GOC_SEMANTICS_STRICT, 0, r.d, r.a, r.b, r.index),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    EXPECT_EQ(std::memcmp(before, r.data, sizeof(before)), 0);
  }
}

TEST(Swmmac16, Wave64PackedIntermediateNarrowing) {
  for (unsigned op : {2u, 3u})
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned odd = 0; odd < 2; ++odd) {
        WideRegisters r(op);
        // Each DOT4 adds only one eighth of an output ULP at 1.0.
        // Narrowing each step discards it even when C has an odd significand.
        uint32_t a = op == 2 ? 0x0200 : 0x3980;
        uint32_t one = op == 2 ? 0x3c00 : 0x3f80;
        for (unsigned reg = 0; reg < 4; ++reg)
          for (auto &word : r.data[reg])
            word = a | (a << 16);
        for (unsigned reg = 4; reg < 12; ++reg)
          for (auto &word : r.data[reg])
            word = one | (one << 16);
        uint32_t c = one + odd;
        for (unsigned reg = 12; reg < 16; ++reg)
          for (auto &word : r.data[reg])
            word = c | (c << 16);
        ASSERT_EQ(wide_functions[op](cpu, 0, r.d, r.a, r.b, r.index), GOC_SUCCESS);
        uint32_t want = one + odd;
        for (unsigned reg = 0; reg < 2; ++reg)
          for (unsigned lane = 0; lane < 64; ++lane)
            EXPECT_EQ(r.d[reg][lane], want | (want << 16));
      }
}

TEST(Swmmac16, Wave64FiniteOverflowAndInfinity) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (bool saturate : {false, true})
      for (bool infinity : {false, true})
        for (bool negative : {false, true}) {
          WideRegisters r(2);
          uint32_t av = infinity ? 0x7c00 : 0x5c00; // Infinity or 256.
          uint32_t bv = infinity ? 0x3c00 : 0x5c00;
          for (unsigned reg = 0; reg < 4; ++reg)
            for (auto &word : r.data[reg])
              word = av | (av << 16);
          for (unsigned reg = 4; reg < 12; ++reg)
            for (auto &word : r.data[reg])
              word = bv | (bv << 16);
          for (unsigned reg = 12; reg < 16; ++reg)
            for (auto &word : r.data[reg])
              word = 0;
          uint32_t modifiers = negative ? GOC_WMMA_NEG_LO_A | GOC_WMMA_NEG_HI_A : 0;
          ASSERT_EQ(wide_functions[2](cpu | (saturate ? GOC_FP16_OVFL : 0), modifiers, r.d, r.a,
                                      r.b, r.index),
                    GOC_SUCCESS);
          uint32_t want = (saturate && !infinity ? 0x7bff : 0x7c00) | (negative ? 0x8000 : 0);
          for (unsigned reg = 0; reg < 2; ++reg)
            for (unsigned lane = 0; lane < 64; ++lane)
              EXPECT_EQ(r.d[reg][lane], want | (want << 16));
        }
}

TEST(Swmmac16, Wave64AccumulatorAndPairNegationAcrossOutputFormats) {
  for (unsigned op = 0; op < 4; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < 3; ++variant) {
        WideRegisters r(op);
        uint32_t one = op & 1 ? 0x3f803f80 : 0x3c003c00;
        for (unsigned reg = 0; reg < 12; ++reg)
          for (unsigned lane = 0; lane < 64; ++lane)
            r.data[reg][lane] = one;
        for (unsigned reg = 12; reg < 20; ++reg)
          for (unsigned lane = 0; lane < 64; ++lane)
            r.data[reg][lane] = op >= 2 ? 0x40004000 : 0x40000000; // Initial accumulator: 2.
        for (unsigned lane = 0; lane < 64; ++lane)
          r.data[20][lane] = 0x44444444; // Select increasing positions 0 and 1.
        uint32_t flags = variant == 0   ? 0
                         : variant == 1 ? GOC_WMMA_NEG_LO_B
                                        : GOC_WMMA_NEG_LO_B | GOC_WMMA_NEG_HI_B;
        ASSERT_EQ(wide_functions[op](cpu, flags, r.d, r.a, r.b, r.index), GOC_SUCCESS);
        // Sixteen products of one: 2+16, 2+8-8, or 2-16.
        const uint32_t fp32[] = {0x41900000, 0x40000000, 0xc1600000};
        const uint32_t fp16[] = {0x4c80, 0x4000, 0xcb00};
        uint32_t expected = op < 2    ? fp32[variant]
                            : op == 2 ? fp16[variant] * 0x10001u
                                      : (fp32[variant] >> 16) * 0x10001u;
        for (unsigned reg = 0; reg < (op >= 2 ? 2u : 4u); ++reg)
          for (unsigned lane = 0; lane < 64; ++lane)
            ASSERT_EQ(r.d[reg][lane], expected) << op << '/' << cpu << '/' << variant;
      }
}

TEST(Swmmac16, FractionalMatricesAgainstDoubleReference) {
  // Logical sparse matrices and double accumulation are independent of the
  // production readers/kernels. Packed outputs are compared as real numbers;
  // packed references narrow mathematically after each four-product group.
  // They model the supported loose contract, not arbitrary empirical exact bits.
  double maximum_error[4] = {};
  for (unsigned seed : {38671U, 97211U})
    for (unsigned width : {32U, 64U})
      for (unsigned op = 0; op < 4; ++op) {
        std::mt19937 random(seed);
        const bool bf16 = op & 1, packed = op >= 2;
        const unsigned mantissa = bf16 ? 7 : 10, bias = bf16 ? 127 : 15;
        auto decode = [&](uint16_t code) {
          unsigned fraction = code & ((1U << mantissa) - 1);
          unsigned exponent = (code & 32767) >> mantissa;
          double value = std::ldexp(double(fraction + (exponent ? 1U << mantissa : 0)),
                                    int(exponent ? exponent : 1) - int(bias) - int(mantissa));
          return code & 32768 ? -value : value;
        };
        auto operand = [&]() {
          unsigned bits = random();
          // Eight exponents, normal finite factors below four, both signs.
          return uint16_t((bits & 32768) | ((bias - 6 + (bits >> 16) % 8) << mantissa) |
                          (bits & ((1U << mantissa) - 1)));
        };
        for (unsigned batch = 0; batch < 64; ++batch)
          for (unsigned key = 0; key < width / 16; ++key) {
            unsigned modifiers = mode((batch % 16) | (key << 4));
            uint32_t original[21][64] = {};
            double a_values[16][16], b_values[32][16], sums[16][16];
            unsigned selected[16][16];
            for (unsigned row = 0; row < 16; ++row)
              for (unsigned pair = 0; pair < 8; ++pair) {
                const unsigned pairs[] = {4, 8, 12, 9, 13, 14};
                unsigned selection = pairs[random() % 6];
                for (unsigned part = 0; part < 2; ++part) {
                  unsigned ck = 2 * pair + part;
                  unsigned lane = row + 16 * (width == 64 ? pair / 2 : pair / 2 % 2);
                  unsigned reg = width == 64 ? pair % 2 : 2 * (pair / 4) + pair % 2;
                  unsigned local = 2 * reg + part;
                  selected[row][ck] = 4 * pair + ((selection >> (2 * part)) & 3);
                  original[20][lane] |= ((selection >> (2 * part)) & 3)
                                        << (key * (width == 64 ? 8 : 16) + 2 * local);
                  uint16_t code = operand();
                  a_values[row][ck] = decode(code);
                  if (modifiers & (part ? GOC_WMMA_NEG_HI_A : GOC_WMMA_NEG_LO_A))
                    a_values[row][ck] = -a_values[row][ck];
                  original[reg][lane] |= uint32_t(code) << (16 * part);
                }
              }
            for (unsigned k = 0; k < 32; ++k)
              for (unsigned col = 0; col < 16; ++col) {
                uint16_t code = operand();
                b_values[k][col] = decode(code);
                unsigned lane = col + 16 * (width == 64 ? k / 8 : k / 8 % 2);
                unsigned reg = width == 64 ? k % 8 / 2 : 4 * (k / 16) + k % 8 / 2;
                original[4 + reg][lane] |= uint32_t(code) << (16 * (k % 2));
              }
            for (unsigned row = 0; row < 16; ++row)
              for (unsigned col = 0; col < 16; ++col) {
                unsigned count = (packed ? 128 : 256) / width;
                unsigned section = packed ? row % 8 / 2 : row % 8;
                unsigned reg = section % count;
                unsigned lane = col + 16 * (row / 8) + 32 * (section / count);
                uint16_t code = operand();
                double sum = decode(code);
                if (packed) {
                  original[12 + reg][lane] |= uint32_t(code) << (16 * (row % 2));
                } else {
                  float c = float(sum);
                  std::memcpy(&original[12 + reg][lane], &c, sizeof(c));
                }
                for (unsigned ck = 0; ck < 16; ++ck) {
                  double b = b_values[selected[row][ck]][col];
                  // B negation follows compressed pair position, not dense K parity.
                  if (modifiers & (ck % 2 ? GOC_WMMA_NEG_HI_B : GOC_WMMA_NEG_LO_B))
                    b = -b;
                  sum += a_values[row][ck] * b;
                  // Model the loose FP32 accumulator before packed-stage rounding.
                  // In this bounded domain, double represents the fused sum exactly.
                  if (packed)
                    sum = double(float(sum));
                  if (packed && (ck + 1) % 4 == 0 && sum != 0) {
                    // BF16 truncates an FP32 intermediate; FP16 rounds to nearest
                    // even. Quantize numerically, independently of production bits.
                    if (bf16)
                      sum = double(float(sum));
                    int exponent;
                    double magnitude = std::frexp(std::abs(sum), &exponent);
                    (void)magnitude;
                    double step = std::ldexp(
                        1.0, std::max(exponent - 1 - int(mantissa), 1 - int(bias) - int(mantissa)));
                    double scaled = std::abs(sum) / step;
                    double integral = std::floor(scaled);
                    if (!bf16 && (scaled - integral > 0.5 ||
                                  (scaled - integral == 0.5 && std::fmod(integral, 2.0) != 0)))
                      ++integral;
                    sum = std::copysign(integral * step, sum);
                  }
                }
                sums[row][col] = sum;
              }
            for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
              uint32_t data[21][64];
              std::memcpy(data, original, sizeof(data));
              const uint32_t *a[] = {data[0], data[1], data[2], data[3]};
              const uint32_t *b[] = {data[4], data[5], data[6],  data[7],
                                     data[8], data[9], data[10], data[11]};
              const uint32_t *index[] = {data[20]};
              uint32_t *d[] = {data[12], data[13], data[14], data[15],
                               data[16], data[17], data[18], data[19]};
              ASSERT_EQ((width == 64 ? wide_functions[op] : functions[op])(cpu, modifiers, d, a, b,
                                                                           index),
                        GOC_SUCCESS);
              unsigned count = (packed ? 128 : 256) / width;
              for (unsigned row = 0; row < 16; ++row)
                for (unsigned col = 0; col < 16; ++col) {
                  unsigned section = packed ? row % 8 / 2 : row % 8;
                  unsigned lane = col + 16 * (row / 8) + 32 * (section / count);
                  uint32_t word = d[section % count][lane];
                  double actual;
                  if (packed) {
                    actual = decode(uint16_t(word >> (16 * (row % 2))));
                  } else {
                    float value;
                    std::memcpy(&value, &word, sizeof(value));
                    actual = value;
                  }
                  auto tolerance = !packed ? goc_test::wmma_f32_tolerance
                                   : bf16  ? goc_test::swmmac_bf16_tolerance
                                           : goc_test::swmmac_f16_tolerance;
                  maximum_error[op] =
                      std::max(maximum_error[op], std::abs(actual - sums[row][col]));
                  EXPECT_TRUE(goc_test::fuzzy(actual, sums[row][col], tolerance))
                      << width << '/' << op << '/' << batch << '/' << key << '/' << cpu
                      << " row=" << row << " col=" << col;
                }
              for (unsigned reg = 0; reg < 21; ++reg)
                for (unsigned lane = 0; lane < 64; ++lane)
                  if (reg < 12 || reg >= 12 + count || lane >= width) {
                    EXPECT_EQ(data[reg][lane], original[reg][lane]);
                  }
            }
          }
      }
  for (unsigned op = 0; op < 4; ++op) {
    std::ostringstream value;
    value << std::setprecision(17) << maximum_error[op];
    RecordProperty("maximum_absolute_error_" + std::to_string(op), value.str());
  }
}

TEST(Swmmac16, CapturedMatricesAndOperands) {
  struct Shape {
    bool bf16, packed;
    decltype(&goc_v_swmmac_f32_16x16x32_f16) wave32, wave64;
    const RdnaWmmaFloatMatrixCase *matrices;
    unsigned matrix_count;
    const RdnaWmmaFloatOperandCase *operands;
    unsigned operand_count;
  };

  const Shape shapes[] = {
      {false, false, goc_v_swmmac_f32_16x16x32_f16, goc_v_swmmac_f32_16x16x32_f16_wave64,
       kRdnaSwmmacF32F16Matrices.data(), kRdnaSwmmacF32F16Matrices.size(),
       kRdnaSwmmacF32F16Operands.data(), kRdnaSwmmacF32F16Operands.size()},
      {true, false, goc_v_swmmac_f32_16x16x32_bf16, goc_v_swmmac_f32_16x16x32_bf16_wave64,
       kRdnaSwmmacF32Bf16Matrices.data(), kRdnaSwmmacF32Bf16Matrices.size(),
       kRdnaSwmmacF32Bf16Operands.data(), kRdnaSwmmacF32Bf16Operands.size()},
      {false, true, goc_v_swmmac_f16_16x16x32_f16, goc_v_swmmac_f16_16x16x32_f16_wave64,
       kRdnaSwmmacF16F16Matrices.data(), kRdnaSwmmacF16F16Matrices.size(),
       kRdnaSwmmacF16F16Operands.data(), kRdnaSwmmacF16F16Operands.size()},
      {true, true, goc_v_swmmac_bf16_16x16x32_bf16, goc_v_swmmac_bf16_16x16x32_bf16_wave64,
       kRdnaSwmmacBf16Bf16Matrices.data(), kRdnaSwmmacBf16Bf16Matrices.size(),
       kRdnaSwmmacBf16Bf16Operands.data(), kRdnaSwmmacBf16Bf16Operands.size()},
  };
  unsigned replayed = 0, excluded = 0;
  for (const auto &shape : shapes)
    for (unsigned sample = 0; sample < shape.matrix_count + shape.operand_count; ++sample)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
        const bool matrix = sample < shape.matrix_count;
        const auto &m = shape.matrices[matrix ? sample : 0];
        const auto &o = shape.operands[matrix ? 0 : sample - shape.matrix_count];
        const unsigned width = matrix ? m.width : o.width;
        const auto code = matrix ? m.code : o.code;
        // Raw ignored sign/clamp encodings and out-of-range key aliases
        // are outside GoC's supported instruction flags, not successful replays.
        unsigned key = (code[0] >> 11) & 7;
        if ((code[0] & 0x8400U) || (code[1] & (1U << 31)) || key >= (width == 64 ? 4U : 2U)) {
          if (cpu == 0)
            ++excluded;
          continue;
        }
        if (cpu == 0)
          ++replayed;
        SCOPED_TRACE(::testing::Message()
                     << shape.bf16 << shape.packed << "/" << sample << "/" << width << "/" << cpu);
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
                const unsigned max_finite = shape.bf16 ? 0x7f7f : 0x7bff;
                value = 0;
                for (unsigned shift = 0; shift < 32; shift += 16) {
                  unsigned part = (rng >> shift) & 65535;
                  value |= ((part & 32768) | (m.bounded_bf16 ? 0x3c80U + (part & 32767) % 0x500U
                                                             : (part & 32767) % (max_finite + 1)))
                           << shift;
                }
              } else if (m.finite && reg < 32) {
                value = accumulators[(rng >> 24) & 7];
                if (m.bounded_bf16 && shape.packed) {
                  constexpr uint32_t packed_c[] = {0,           0x00010001U, 0x80008000U,
                                                   0x3f813f81U, 0xbf81bf81U, 0x43814381U,
                                                   0xc381c381U, 0x80018001U};
                  value = packed_c[(rng >> 24) & 7];
                }
              }
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
        unsigned count = matrix ? m.expected_words : shape.packed ? 128 : 256;
        for (unsigned i = 0; i < count; ++i)
          expected[dest + i / width][1 + i % width] = matrix ? m.expected[i] : o.expected;
        unsigned a = (code[1] & 511) - 256, b = ((code[1] >> 9) & 511) - 256,
                 index = ((code[1] >> 18) & 511) - 256;
        ASSERT_LT(a, 33U);
        ASSERT_LT(b, 33U);
        ASSERT_LT(index, 33U);
        ASSERT_EQ((width == 32 ? shape.wave32 : shape.wave64)(
                      cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT |
                          (((matrix ? m.mode : o.mode) & (1U << 23)) ? GOC_FP16_OVFL : 0),
                      (uint64_t(key) << 7) | ((code[1] >> 29) & 3) | (((code[0] >> 8) & 3) << 3),
                      v + dest, v + a, v + b, v + index),
                  GOC_SUCCESS);
        for (unsigned reg = 0; reg < 33; ++reg)
          for (unsigned lane = 0; lane < 66; ++lane)
            ASSERT_EQ(data[reg][lane], expected[reg][lane])
                << "reg=" << reg << " guarded lane=" << lane;
      }
  EXPECT_EQ(replayed, 488U);
  EXPECT_EQ(excluded, 24U);
}

TEST(Swmmac16, PackedIntermediateOverflowCannotCancel) {
  for (bool wave64 : {false, true})
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool saturate : {false, true}) {
        uint32_t a_data[4][64] = {}, b_data[8][64], d_data[4][64] = {}, indices[64];
        const uint32_t *a[] = {a_data[0], a_data[1], a_data[2], a_data[3]};
        const uint32_t *b[] = {b_data[0], b_data[1], b_data[2], b_data[3],
                               b_data[4], b_data[5], b_data[6], b_data[7]};
        uint32_t *d[] = {d_data[0], d_data[1], d_data[2], d_data[3]};
        const uint32_t *index[] = {indices};
        std::fill(indices, indices + 64, 0x44444444U);
        for (auto &reg : b_data)
          std::fill(reg, reg + 64, 0x5c005c00U); // +256
        // +256 in the first DOT4, -256 in the second, then zeros.
        for (unsigned row = 0; row < 16; ++row)
          for (unsigned ck = 0; ck < 8; ++ck) {
            unsigned lane = wave64 ? row + 16 * (ck / 4) : row + 16 * ((ck / 4) & 1);
            unsigned reg = wave64 ? (ck % 4) / 2 : (ck / 2) & 1;
            a_data[reg][lane] |= uint32_t(ck < 4 ? 0x5c00 : 0xdc00) << (16 * (ck % 2));
          }
        auto fn = wave64 ? goc_v_swmmac_f16_16x16x32_f16_wave64 : goc_v_swmmac_f16_16x16x32_f16;
        ASSERT_EQ(fn(cpu | (saturate ? GOC_FP16_OVFL : 0), 0, d, a, b, index), GOC_SUCCESS);
        uint32_t half = saturate ? 0xfbff : 0x7c00;
        for (unsigned reg = 0; reg < (wave64 ? 2U : 4U); ++reg)
          for (unsigned lane = 0; lane < (wave64 ? 64U : 32U); ++lane)
            EXPECT_EQ(d_data[reg][lane], half | (half << 16));
      }
}

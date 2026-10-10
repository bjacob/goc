// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "goc/goc.h"
#include "swmmac16_hardware.h"
#include "swmmac16_wave64_hardware.h"

#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>

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
  for (unsigned op = 0; op < 4; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < 32; ++variant) {
        Registers r(op);
        ASSERT_EQ(functions[op](cpu, mode(variant), r.d, r.a, r.b, r.index), GOC_SUCCESS);
        uint64_t digest = goc_test::capture_hash_seed;
        for (unsigned reg = 0; reg < (op >= 2 ? 4u : 8u); ++reg)
          for (unsigned lane = 0; lane < 32; ++lane)
            digest = goc_test::capture_hash_bytes(digest, r.d[reg][lane], 4);
        ASSERT_EQ(digest, goc_test::swmmac16_capture_digests[op][variant])
            << op << "/" << cpu << "/" << variant;
      }
}

TEST(Swmmac16, EveryLaneAndModifier) {
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
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, nullptr, nullptr, nullptr,
                 nullptr),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    for (unsigned bit = 0; bit < 32; ++bit)
      if (!((1u << bit) & mode(31))) {
        EXPECT_EQ(fn(0, 1u << bit, nullptr, nullptr, nullptr, nullptr), GOC_ERROR_INVALID_FLAGS);
      }
  }
}

TEST(Swmmac16, PackedNearestEvenMidpoints) {
  for (unsigned op : {2u, 3u})
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned odd = 0; odd < 2; ++odd) {
        Registers r(op);
        // Sixteen equal products add exactly half an output ULP at 1.0.
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
        uint32_t want = one + 2 * odd;
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
  for (unsigned op = 0; op < 4; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < 64; ++variant) {
        uint32_t data[21][64];
        goc_test::swmmac16_capture_inputs(op, data);
        const uint32_t *a[] = {data[0], data[1]}, *b[] = {data[4], data[5], data[6], data[7]},
                       *index[] = {data[20]};
        uint32_t *d[] = {data[12], data[13], data[14], data[15]};
        ASSERT_EQ(wide_functions[op](cpu, mode(variant), d, a, b, index), GOC_SUCCESS);
        uint64_t hash = goc_test::capture_hash_seed;
        for (unsigned reg = 0; reg < (op >= 2 ? 2U : 4U); ++reg)
          for (unsigned lane = 0; lane < 64; ++lane)
            hash = goc_test::capture_hash_word(hash, d[reg][lane]);
        ASSERT_EQ(hash, goc_test::swmmac16_wave64_digests[op][variant])
            << op << "/" << cpu << "/" << variant;
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
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, nullptr, nullptr, nullptr,
                 nullptr),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    for (unsigned bit = 0; bit < 64; ++bit)
      if (!((1ULL << bit) & mode(63))) {
        EXPECT_EQ(fn(0, 1ULL << bit, nullptr, nullptr, nullptr, nullptr), GOC_ERROR_INVALID_FLAGS);
      }
  }
}

TEST(Swmmac16, Wave64PackedNearestEvenMidpoints) {
  for (unsigned op : {2u, 3u})
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned odd = 0; odd < 2; ++odd) {
        WideRegisters r(op);
        // Sixteen equal products add exactly half an output ULP at 1.0.
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
        uint32_t want = one + 2 * odd;
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

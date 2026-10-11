// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "goc/goc.h"
#include "integer_wmma_capture.h"
#include "rdna_swmmac_iu4_cases.h"
#include "rdna_swmmac_iu4_k64_cases.h"
#include "rdna_swmmac_iu8_cases.h"
#include "swmmac_integer_hardware.h"
#include "swmmac_integer_wave64_hardware.h"

#include <algorithm>
#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_v_swmmac_i32_16x16x32_iu8);
const Fn functions[] = {goc_v_swmmac_i32_16x16x32_iu8, goc_v_swmmac_i32_16x16x32_iu4,
                        goc_v_swmmac_i32_16x16x64_iu4};

uint32_t mode(unsigned variant) {
  return (variant & 3) | ((variant & 4) << 4) | ((variant >> 3) << 7);
}

template <unsigned Lanes> struct LaneRegisters {
  static const unsigned a_count = Lanes == 64 ? 1 : 2;
  static const unsigned b_count = Lanes == 64 ? 2 : 4;
  static const unsigned d_count = Lanes == 64 ? 4 : 8;
  uint32_t data[15][Lanes];
  const uint32_t *a[a_count], *b[b_count], *index[1];
  uint32_t *d[d_count];

  explicit LaneRegisters(unsigned /*op*/, unsigned sample = 0) {
    goc_test::integer_wmma_capture_inputs(sample, data);
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

TEST(SwmmacInteger, CaptureInputsKeepSeedAndDrawOrder) {
  const uint64_t expected[] = {0x7689d9af055e43b6ULL, 0x1fadfb09a98d158fULL};
  for (unsigned i = 0; i < 2; ++i) {
    uint32_t data[15][32];
    goc_test::integer_wmma_capture_inputs(i ? 15 : 0, data);
    uint64_t hash = goc_test::capture_hash_seed;
    for (const auto &reg : data)
      for (uint32_t word : reg)
        hash = goc_test::capture_hash_bytes(hash, word, 4);
    EXPECT_EQ(hash, expected[i]);
  }
}

TEST(SwmmacInteger, HardwareAllModifiers) {
  for (unsigned op = 0; op < 3; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < (op == 2 ? 8u : 16u); ++variant)
        for (unsigned sample = 0; sample < 16; ++sample)
          for (uint64_t semantics :
               {uint64_t{0}, GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT}) {
            Registers r(op, sample);
            uint32_t before[15][32];
            std::memcpy(before, r.data, sizeof(before));
            ASSERT_EQ(functions[op](cpu | semantics, mode(variant), r.d, r.a, r.b, r.index),
                      GOC_SUCCESS);
            for (unsigned reg = 0; reg < 15; ++reg) {
              if (reg < 6 || reg >= 14) {
                ASSERT_EQ(std::memcmp(r.data[reg], before[reg], sizeof(before[reg])), 0)
                    << "source=" << reg;
              }
            }
            uint64_t digest = goc_test::capture_hash_seed;
            for (unsigned reg = 0; reg < 8u; ++reg)
              for (unsigned lane = 0; lane < 32; ++lane)
                digest = goc_test::capture_hash_bytes(digest, r.d[reg][lane], 4);
            ASSERT_EQ(digest, goc_test::swmmac_integer_capture_digests[op][sample][variant])
                << op << "/" << cpu << "/" << sample << "/" << variant << "/" << semantics;
          }
}

// Differential CPU dispatch check; the independent hardware digests above
// establish numerical behavior, while this test localizes per-register failures.
TEST(SwmmacInteger, EveryLaneAndModifier) {
  for (unsigned op = 0; op < 3; ++op)
    for (unsigned variant = 0; variant < (op == 2 ? 8u : 16u); ++variant) {
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

TEST(SwmmacInteger, OverlappingSourcesAndDuplicateDestinations) {
  for (unsigned op = 0; op < 3; ++op)
    for (unsigned variant = 0; variant < (op == 2 ? 8u : 16u); ++variant)
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

TEST(SwmmacInteger, Validation) {
  for (unsigned op = 0; op < 3; ++op) {
    Registers r(op);
    uint32_t before[15][32];
    std::memcpy(before, r.data, sizeof(before));
    uint64_t known = mode(op == 2 ? 7 : 15);
    for (unsigned bit = 0; bit < 64; ++bit) {
      if (!(known & (1ULL << bit))) {
        EXPECT_EQ(functions[op](0, 1ULL << bit, r.d, r.a, r.b, r.index), GOC_ERROR_INVALID_FLAGS);
      }
    }
    EXPECT_EQ(functions[op](1ULL << 63, 0, r.d, r.a, r.b, r.index), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(std::memcmp(before, r.data, sizeof(before)), 0);
  }
}

TEST(SwmmacInteger, ClampSaturatesEachHardwareStage) {
  for (unsigned op = 0; op < 3; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool negative_first : {false, true})
        for (bool clamp : {false, true}) {
          Registers r(op);
          const uint32_t ones = op == 0 ? 0x01010101 : 0x11111111;
          const uint32_t initial = negative_first ? 0x80000000 : 0x7fffffff;
          // The two lane halves supply opposite product signs. The full sum
          // is zero, but the first stage saturates before the second undoes it.
          for (unsigned reg = 0; reg < 2; ++reg)
            for (unsigned lane = 0; lane < 32; ++lane)
              r.data[reg][lane] = ((lane >= 16) != negative_first) ? UINT32_MAX : ones;
          for (unsigned reg = 2; reg < 6; ++reg)
            for (auto &word : r.data[reg])
              word = ones;
          for (unsigned reg = 6; reg < 14; ++reg)
            for (auto &word : r.data[reg])
              word = initial;
          uint32_t modifiers = GOC_WMMA_SIGNED_A | (clamp ? GOC_WMMA_CLAMP : 0);
          ASSERT_EQ(functions[op](cpu, modifiers, r.d, r.a, r.b, r.index), GOC_SUCCESS);
          unsigned stage_products = op == 2 ? 16 : 8;
          uint32_t want = initial;
          if (clamp)
            want = negative_first ? initial + stage_products : initial - stage_products;
          for (unsigned reg = 0; reg < 8; ++reg)
            for (unsigned lane = 0; lane < 32; ++lane)
              EXPECT_EQ(r.d[reg][lane], want);
        }
}

namespace {

const Fn wide_functions[] = {goc_v_swmmac_i32_16x16x32_iu8_wave64,
                             goc_v_swmmac_i32_16x16x32_iu4_wave64,
                             goc_v_swmmac_i32_16x16x64_iu4_wave64};

} // namespace

TEST(SwmmacInteger, Wave64NativeAllModifiers) {
  for (unsigned op = 0; op < 3; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < (op == 0 ? 32U : 16U); ++variant)
        for (unsigned sample = 0; sample < 16; ++sample)
          for (uint64_t semantics :
               {uint64_t(0), GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT}) {
            uint32_t data[15][64], before[15][64];
            goc_test::integer_wmma_capture_inputs(sample, data);
            std::memcpy(before, data, sizeof(before));
            const uint32_t *a[] = {data[0]}, *b[] = {data[2], data[3]}, *index[] = {data[14]};
            uint32_t *d[] = {data[6], data[7], data[8], data[9]};
            unsigned modifiers = (variant & 3) | ((variant & 4) << 4) | ((variant >> 3) << 7);
            ASSERT_EQ(wide_functions[op](cpu | semantics, modifiers, d, a, b, index), GOC_SUCCESS);
            for (unsigned reg = 0; reg < 15; ++reg) {
              if (reg < 6 || reg >= 10) {
                ASSERT_EQ(std::memcmp(data[reg], before[reg], sizeof(before[reg])), 0)
                    << "source=" << reg;
              }
            }
            uint64_t hash = goc_test::capture_hash_seed;
            for (unsigned reg = 0; reg < 4; ++reg)
              for (unsigned lane = 0; lane < 64; ++lane)
                hash = goc_test::capture_hash_word(hash, d[reg][lane]);
            ASSERT_EQ(hash, goc_test::swmmac_integer_wave64_digests[op][sample][variant])
                << op << "/" << cpu << "/" << sample << "/" << variant << "/" << semantics;
          }
}

TEST(SwmmacInteger, Wave64OverlappingSourcesAndDuplicateDestinations) {
  for (unsigned op = 0; op < 3; ++op)
    for (unsigned variant = 0; variant < (op == 0 ? 32u : 16u); ++variant)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (unsigned alias = 0; alias < 4; ++alias) {
          WideRegisters r(op), reference(op);
          unsigned regs = 4;
          unsigned offsets[4];
          for (unsigned reg = 0; reg < regs; ++reg) {
            offsets[reg] = alias == 0   ? 0
                           : alias == 1 ? 2 + reg % (op == 1 ? 1 : 2)
                           : alias == 2 ? 14
                                        : 6;
            r.d[reg] = r.data[offsets[reg]];
            std::memcpy(reference.d[reg], r.d[reg], 64 * sizeof(uint32_t));
          }

          ASSERT_EQ(wide_functions[op](0, mode(variant), reference.d, reference.a, reference.b,
                                       reference.index),
                    GOC_SUCCESS);
          uint32_t expected[15][64];
          std::memcpy(expected, r.data, sizeof(expected));
          for (unsigned reg = 0; reg < regs; ++reg)
            for (unsigned lane = 0; lane < 64; ++lane)
              expected[offsets[reg]][lane] = reference.d[reg][lane];
          ASSERT_EQ(wide_functions[op](cpu, mode(variant), r.d, r.a, r.b, r.index), GOC_SUCCESS);
          ASSERT_EQ(std::memcmp(expected, r.data, sizeof(expected)), 0)
              << op << "/" << cpu << "/" << variant << "/" << alias;
        }
}

TEST(SwmmacInteger, Wave64ClampSaturatesEachHardwareStage) {
  for (unsigned op = 0; op < 3; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool negative_first : {false, true})
        for (bool clamp : {false, true}) {
          WideRegisters r(op);
          const uint32_t ones = op == 0 ? 0x01010101 : 0x11111111;
          const uint32_t initial = negative_first ? 0x80000000 : 0x7fffffff;
          // The two lane halves supply opposite product signs. The full sum
          // is zero, but the first stage saturates before the second undoes it.
          for (unsigned reg = 0; reg < 2; ++reg)
            for (unsigned lane = 0; lane < 64; ++lane)
              r.data[reg][lane] = (bool(lane & 16) != negative_first) ? UINT32_MAX : ones;
          for (unsigned reg = 2; reg < 6; ++reg)
            for (auto &word : r.data[reg])
              word = ones;
          for (unsigned reg = 6; reg < 14; ++reg)
            for (auto &word : r.data[reg])
              word = initial;
          uint32_t modifiers = GOC_WMMA_SIGNED_A | (clamp ? GOC_WMMA_CLAMP : 0);
          ASSERT_EQ(wide_functions[op](cpu, modifiers, r.d, r.a, r.b, r.index), GOC_SUCCESS);
          unsigned stage_products = op == 2 ? 16 : 8;
          uint32_t want = initial;
          if (clamp)
            want = negative_first ? initial + stage_products : initial - stage_products;
          for (unsigned reg = 0; reg < 4; ++reg)
            for (unsigned lane = 0; lane < 64; ++lane)
              EXPECT_EQ(r.d[reg][lane], want);
        }
}

TEST(SwmmacInteger, Wave64ValidationPreservesOutputs) {
  for (unsigned op = 0; op < 3; ++op) {
    WideRegisters r(op);
    uint32_t before[15][64];
    std::memcpy(before, r.data, sizeof before);
    uint64_t known = mode(op == 0 ? 31 : 15);
    for (unsigned bit = 0; bit < 64; ++bit)
      if (!(known & (1ULL << bit))) {
        EXPECT_EQ(wide_functions[op](0, 1ULL << bit, r.d, r.a, r.b, r.index),
                  GOC_ERROR_INVALID_FLAGS);
      }
    EXPECT_EQ(wide_functions[op](1ULL << 63, 0, r.d, r.a, r.b, r.index), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(std::memcmp(before, r.data, sizeof before), 0);
  }
}

// Captures use tied C/D and include A/B/index destination overlaps. Recreate
// rocjitsu's unsigned xorshift draw order, including the metadata register.
TEST(SwmmacInteger, JakubCapturedSaturationKeysAndOverlap) {
  struct Shape {
    Fn wave32, wave64;
    const RdnaSwmmacIntegerCase *cases;
    unsigned count;
  };

  const Shape shapes[] = {
      {goc_v_swmmac_i32_16x16x32_iu8, goc_v_swmmac_i32_16x16x32_iu8_wave64,
       kRdnaSwmmacIu8Saturated.data(), kRdnaSwmmacIu8Saturated.size()},
      {goc_v_swmmac_i32_16x16x32_iu4, goc_v_swmmac_i32_16x16x32_iu4_wave64,
       kRdnaSwmmacIu4Saturated.data(), kRdnaSwmmacIu4Saturated.size()},
      {goc_v_swmmac_i32_16x16x64_iu4, goc_v_swmmac_i32_16x16x64_iu4_wave64,
       kRdnaSwmmacIu4K64Saturated.data(), kRdnaSwmmacIu4K64Saturated.size()},
  };
  for (const auto &shape : shapes)
    for (unsigned sample = 0; sample < shape.count; ++sample) {
      const auto &capture = shape.cases[sample];
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL}) {
          SCOPED_TRACE(::testing::Message()
                       << shape.count << "/" << sample << "/" << capture.width << "/" << capture.key
                       << "/" << capture.dst << "/" << cpu << "/" << semantics);
          uint32_t data[40][66], expected[40][66];
          uint32_t *v[40];
          uint32_t rng = 0x12056101U;
          constexpr uint32_t accumulators[] = {0,           1,           0xffffffffU, 0x7fffffffU,
                                               0x80000000U, 0x7ffff000U, 0x80001000U, 0x12345678U};
          constexpr uint32_t pairs[] = {4, 8, 12, 9, 13, 14};
          for (unsigned reg = 0; reg < 40; ++reg) {
            v[reg] = data[reg] + 1;
            std::fill(data[reg], data[reg] + 66, 0xdeadbeefU);
            if (reg >= 33)
              continue;
            for (unsigned lane = 0; lane < capture.width; ++lane) {
              rng ^= rng << 13;
              rng ^= rng >> 17;
              rng ^= rng << 5;
              uint32_t value = reg >= 16 && reg < 32 ? accumulators[(rng >> 24) & 7] : rng;
              if (reg == 32) {
                value = 0;
                for (unsigned i = 0; i < 8; ++i)
                  value |= pairs[((rng >> (4 * i)) & 15) % 6] << (4 * i);
              }
              v[reg][lane] = value;
            }
          }
          std::memcpy(expected, data, sizeof(expected));
          for (unsigned reg = 0; reg < 256 / capture.width; ++reg)
            for (unsigned lane = 0; lane < capture.width; ++lane)
              expected[capture.dst + reg][lane + 1] = capture.expected[reg * capture.width + lane];
          const auto fn = capture.width == 32 ? shape.wave32 : shape.wave64;
          ASSERT_EQ(fn(cpu | semantics | GOC_SEMANTICS_STRICT,
                       GOC_WMMA_SIGNED_A | GOC_WMMA_SIGNED_B | GOC_WMMA_CLAMP |
                           (uint64_t(capture.key) << 7),
                       v + capture.dst, v, v + 8, v + 32),
                    GOC_SUCCESS);
          for (unsigned reg = 0; reg < 40; ++reg)
            for (unsigned lane = 0; lane < 66; ++lane)
              ASSERT_EQ(data[reg][lane], expected[reg][lane])
                  << "reg=" << reg << " guarded lane=" << lane;
        }
    }
}

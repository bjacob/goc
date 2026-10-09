// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "goc/goc.h"
#include "rdna4_exec_masks.h"
#include "rdna4_integer_wmma_capture.h"
#include "rdna4_swmmac_integer_hardware.h"

#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_rdna4_v_swmmac_i32_16x16x32_iu8);
const Fn functions[] = {goc_rdna4_v_swmmac_i32_16x16x32_iu8, goc_rdna4_v_swmmac_i32_16x16x32_iu4,
                        goc_rdna4_v_swmmac_i32_16x16x64_iu4};

uint32_t mode(unsigned variant) {
  return (variant & 3) | ((variant & 4) << 4) | ((variant & 8) << 4);
}

struct Registers {
  uint32_t data[15][32];
  const uint32_t *a[2], *b[4], *index[1];
  uint32_t *d[8];

  explicit Registers(unsigned /*op*/, unsigned sample = 0) {
    goc_test::integer_wmma_capture_inputs(sample, data);
    for (unsigned reg = 0; reg < 2; ++reg)
      a[reg] = data[reg];
    for (unsigned reg = 0; reg < 4; ++reg)
      b[reg] = data[2 + reg];
    for (unsigned reg = 0; reg < 8; ++reg)
      d[reg] = data[6 + reg];
    index[0] = data[14];
  }
};

} // namespace

TEST(SwmmacInteger, CaptureInputsKeepSeedAndDrawOrder) {
  const uint64_t expected[] = {UINT64_C(0x7689d9af055e43b6), UINT64_C(0x1fadfb09a98d158f)};
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
               {UINT64_C(0), GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT}) {
            Registers r(op, sample);
            ASSERT_EQ(
                functions[op](cpu | semantics, UINT32_MAX, mode(variant), r.d, r.a, r.b, r.index),
                GOC_SUCCESS);
            uint64_t digest = goc_test::capture_hash_seed;
            for (unsigned reg = 0; reg < 8u; ++reg)
              for (unsigned lane = 0; lane < 32; ++lane)
                digest = goc_test::capture_hash_bytes(digest, r.d[reg][lane], 4);
            ASSERT_EQ(digest, goc_test::swmmac_integer_capture_digests[op][sample][variant])
                << op << "/" << cpu << "/" << variant;
          }
}

TEST(SwmmacInteger, EveryExecLaneAndModifier) {
  for (unsigned op = 0; op < 3; ++op)
    for (unsigned variant = 0; variant < (op == 2 ? 8u : 16u); ++variant) {
      Registers full(op), initial(op);
      ASSERT_EQ(functions[op](0, UINT32_MAX, mode(variant), full.d, full.a, full.b, full.index),
                GOC_SUCCESS);
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (uint32_t mask : rdna4_exec_masks()) {
          Registers r(op);
          ASSERT_EQ(functions[op](cpu, mask, mode(variant), r.d, r.a, r.b, r.index), GOC_SUCCESS);
          for (unsigned reg = 0; reg < 8u; ++reg)
            for (unsigned lane = 0; lane < 32; ++lane)
              ASSERT_EQ(r.d[reg][lane],
                        ((mask >> lane) & 1) ? full.d[reg][lane] : initial.d[reg][lane])
                  << op << "/" << cpu << "/" << variant << "/" << mask;
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
          const uint32_t mask = 0xa35ac69d;
          ASSERT_EQ(functions[op](0, mask, mode(variant), reference.d, reference.a, reference.b,
                                  reference.index),
                    GOC_SUCCESS);
          uint32_t expected[15][32];
          std::memcpy(expected, r.data, sizeof(expected));
          for (unsigned reg = 0; reg < regs; ++reg)
            for (unsigned lane = 0; lane < 32; ++lane)
              if ((mask >> lane) & 1)
                expected[offsets[reg]][lane] = reference.d[reg][lane];
          ASSERT_EQ(functions[op](cpu, mask, mode(variant), r.d, r.a, r.b, r.index), GOC_SUCCESS);
          ASSERT_EQ(std::memcmp(expected, r.data, sizeof(expected)), 0)
              << op << "/" << cpu << "/" << variant << "/" << alias;
        }
}

TEST(SwmmacInteger, ValidationAndEmptyExec) {
  for (unsigned op = 0; op < 3; ++op) {
    Fn fn = functions[op];
    EXPECT_EQ(fn(0, UINT32_C(0), 0, nullptr, nullptr, nullptr, nullptr), GOC_SUCCESS);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0, nullptr, nullptr,
                 nullptr, nullptr),
              GOC_SUCCESS);
    for (unsigned bit = 0; bit < 32; ++bit) {
      if (!((1u << bit) & mode(op == 2 ? 7 : 15))) {
        EXPECT_EQ(fn(0, 0, 1u << bit, nullptr, nullptr, nullptr, nullptr), GOC_ERROR_INVALID_FLAGS);
      }
    }
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
          ASSERT_EQ(functions[op](cpu, UINT32_MAX, modifiers, r.d, r.a, r.b, r.index), GOC_SUCCESS);
          unsigned stage_products = op == 2 ? 16 : 8;
          uint32_t want = initial;
          if (clamp)
            want = negative_first ? initial + stage_products : initial - stage_products;
          for (unsigned reg = 0; reg < 8; ++reg)
            for (unsigned lane = 0; lane < 32; ++lane)
              EXPECT_EQ(r.d[reg][lane], want);
        }
}

// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "goc/goc.h"
#include "rdna4_exec_masks.h"
#include "rdna4_interp32_hardware.h"
#include "rdna4_interp32_reference.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_rdna4_v_interp_p10_f32);
const Fn functions[] = {goc_rdna4_v_interp_p10_f32, goc_rdna4_v_interp_p2_f32};

void hash_words(uint64_t &hash, const uint32_t *words) {
  for (unsigned lane = 0; lane < 32; ++lane)
    hash = goc_test::capture_hash_bytes(hash, words[lane], 4);
}

bool equal_float(uint32_t got, float want) {
  float value = goc_test::interp32_float(got);
  if (std::isnan(want))
    return std::isnan(value);
  if (value == want)
    return true;
  if (!std::isfinite(value) || !std::isfinite(want))
    return false;
  return std::abs(double(value) - want) <= 2e-7 * std::abs(double(want)) + 1e-44;
}

} // namespace

TEST(Interp32, HardwareModifiersAndWaitCounts) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned op = 0; op < 2; ++op)
      for (unsigned m = 0; m < 16; ++m)
        for (unsigned wait = 0; wait < 8; ++wait) {
          uint32_t words[4][32];
          goc_test::interp32_capture_inputs(words);
          const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
          uint32_t *d[] = {words[3]};
          ASSERT_EQ(functions[op](cpu, UINT32_MAX, goc_test::interp32_mode(m, wait), d, a, b, c),
                    GOC_SUCCESS);
          uint64_t digest = goc_test::capture_hash_seed;
          hash_words(digest, words[3]);
          EXPECT_EQ(digest, goc_test::interp32_full_digests[(op * 16 + m) * 8 + wait])
              << cpu << "/" << op << "/" << m << "/" << wait;
        }
}

TEST(Interp32, HardwareReadsInactiveSourceLanes) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned mi = 0; mi < 36; ++mi) {
      const uint32_t first[] = {0, UINT32_MAX, 0x55555555, 0xaaaaaaaa};
      uint32_t mask = mi < 4 ? first[mi] : 1u << (mi - 4);
      uint64_t digest = goc_test::capture_hash_seed;
      for (unsigned op = 0; op < 2; ++op)
        for (unsigned m = 0; m < 16; ++m) {
          uint32_t words[4][32];
          goc_test::interp32_capture_inputs(words);
          const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
          uint32_t *d[] = {words[3]};
          ASSERT_EQ(functions[op](cpu, mask, goc_test::interp32_mode(m), d, a, b, c), GOC_SUCCESS);
          hash_words(digest, words[3]);
        }
      EXPECT_EQ(digest, goc_test::interp32_masked_digests[mi]) << cpu << "/" << mi;
    }
}

TEST(Interp32, MasksModifiersAliasesAndUnalignedStorage) {
  const unsigned sources[][3] = {{0, 1, 2}, {0, 0, 2}, {0, 1, 0}, {0, 1, 1}, {0, 0, 0}};
  for (unsigned op = 0; op < 2; ++op)
    for (unsigned m = 0; m < 16; ++m)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (const auto &source : sources)
          for (unsigned target = 0; target < 4; ++target)
            for (uint32_t mask : rdna4_exec_masks()) {
              uint32_t words[4][35], expected[4][35];
              for (unsigned reg = 0; reg < 4; ++reg)
                for (unsigned lane = 0; lane < 35; ++lane)
                  words[reg][lane] = expected[reg][lane] =
                      goc_test::interp32_bits(float(int(lane * 3 + reg * 17) - 45) * 0.125f);
              for (unsigned lane = 0; lane < 32; ++lane)
                if ((mask >> lane) & 1)
                  expected[target][lane + 1] = goc_test::interp32_bits(
                      goc_test::interp32_reference(op, words[source[0]] + 1, words[source[1]] + 1,
                                                   words[source[2]] + 1, lane, m));
              const uint32_t *a[] = {words[source[0]] + 1}, *b[] = {words[source[1]] + 1},
                             *c[] = {words[source[2]] + 1};
              uint32_t *d[] = {words[target] + 1};
              ASSERT_EQ(functions[op](cpu, mask, goc_test::interp32_mode(m, m % 8), d, a, b, c),
                        GOC_SUCCESS);
              for (unsigned reg = 0; reg < 4; ++reg)
                ASSERT_TRUE(std::equal(words[reg], words[reg] + 35, expected[reg]))
                    << cpu << "/" << op << "/" << m << "/" << target;
            }
}

TEST(Interp32, SpecialValuesAndRandomBits) {
  const uint32_t edges[] = {0,          0x80000000, 0x3f800000, 0xbf800000, 0x7f800000, 0xff800000,
                            0x7fc12345, 0x7f812345, 1,          0x80000001, 0x7f7fffff, 0xff7fffff,
                            0x00800000, 0x80800000, 0x3f000000, 0xbf000000};
  for (unsigned op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned m = 0; m < 16; ++m) {
        std::mt19937 random(971);
        for (unsigned batch = 0; batch < 96; ++batch) {
          uint32_t words[4][32];
          for (unsigned reg = 0; reg < 3; ++reg)
            for (unsigned lane = 0; lane < 32; ++lane)
              words[reg][lane] = batch < 32 ? edges[(batch + lane * (reg + 1) + reg * 7) % 16]
                                            : uint32_t(random());
          const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
          uint32_t *d[] = {words[3]};
          ASSERT_EQ(functions[op](cpu, UINT32_MAX, goc_test::interp32_mode(m), d, a, b, c),
                    GOC_SUCCESS);
          for (unsigned lane = 0; lane < 32; ++lane)
            ASSERT_TRUE(equal_float(words[3][lane], goc_test::interp32_reference(
                                                        op, words[0], words[1], words[2], lane, m)))
                << cpu << "/" << op << "/" << m << "/" << batch << "/" << lane;
        }
      }
}

TEST(Interp32, ValidationAndEmptyExec) {
  uint32_t known = goc_test::interp32_mode(15, 7);
  for (Fn fn : functions) {
    EXPECT_EQ(fn(0, 0U, known, nullptr, nullptr, nullptr, nullptr), GOC_SUCCESS);
    for (unsigned bit = 0; bit < 32; ++bit) {
      if (!(known & (1u << bit))) {
        EXPECT_EQ(fn(0, 0, 1u << bit, nullptr, nullptr, nullptr, nullptr), GOC_ERROR_INVALID_FLAGS);
      }
    }
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0, nullptr, nullptr,
                 nullptr, nullptr),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
  }
}

// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "exec_masks.h"
#include "fuzzy.h"
#include "goc/goc.h"
#include "interp32_hardware.h"
#include "interp32_reference.h"
#include "loose_tolerances.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_v_interp_p10_f32);
const Fn functions[] = {goc_v_interp_p10_f32, goc_v_interp_p2_f32};

void hash_words(uint64_t &hash, const uint32_t *words) {
  for (unsigned lane = 0; lane < 32; ++lane)
    hash = goc_test::capture_hash_bytes(hash, words[lane], 4);
}

::testing::AssertionResult equal_float(uint32_t got, float want) {
  return goc_test::fuzzy(goc_test::interp32_float(got), want, goc_test::interp_f32_tolerance);
}

const decltype(&goc_v_interp_p10_f32_wave64) wave64_functions[] = {goc_v_interp_p10_f32_wave64,
                                                                   goc_v_interp_p2_f32_wave64};

void wave64_capture_inputs(uint32_t words[4][64]) {
  for (unsigned half = 0; half < 2; ++half) {
    uint32_t captured[4][32];
    goc_test::interp32_capture_inputs(captured);
    for (unsigned reg = 0; reg < 4; ++reg)
      std::memcpy(words[reg] + half * 32, captured[reg], sizeof(captured[reg]));
  }
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
      uint32_t exec_mask = mi < 4 ? first[mi] : 1u << (mi - 4);
      uint64_t digest = goc_test::capture_hash_seed;
      for (unsigned op = 0; op < 2; ++op)
        for (unsigned m = 0; m < 16; ++m) {
          uint32_t words[4][32];
          goc_test::interp32_capture_inputs(words);
          const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
          uint32_t *d[] = {words[3]};
          ASSERT_EQ(functions[op](cpu, exec_mask, goc_test::interp32_mode(m), d, a, b, c),
                    GOC_SUCCESS);
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
            for (uint32_t exec_mask : exec_masks()) {
              uint32_t words[4][35], expected[4][35];
              for (unsigned reg = 0; reg < 4; ++reg)
                for (unsigned lane = 0; lane < 35; ++lane)
                  words[reg][lane] = expected[reg][lane] =
                      goc_test::interp32_bits(float(int(lane * 3 + reg * 17) - 45) * 0.125f);
              for (unsigned lane = 0; lane < 32; ++lane)
                if ((exec_mask >> lane) & 1)
                  expected[target][lane + 1] = goc_test::interp32_bits(
                      goc_test::interp32_reference(op, words[source[0]] + 1, words[source[1]] + 1,
                                                   words[source[2]] + 1, lane, m));
              const uint32_t *a[] = {words[source[0]] + 1}, *b[] = {words[source[1]] + 1},
                             *c[] = {words[source[2]] + 1};
              uint32_t *d[] = {words[target] + 1};
              ASSERT_EQ(
                  functions[op](cpu, exec_mask, goc_test::interp32_mode(m, m % 8), d, a, b, c),
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

TEST(Interp32, Wave64HardwareModifiersAndWaitCounts) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned op = 0; op < 2; ++op)
      for (unsigned m = 0; m < 16; ++m)
        for (unsigned wait = 0; wait < 8; ++wait) {
          uint32_t words[4][64];
          wave64_capture_inputs(words);
          const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
          uint32_t *d[] = {words[3]};
          ASSERT_EQ(
              wave64_functions[op](cpu, UINT64_MAX, goc_test::interp32_mode(m, wait), d, a, b, c),
              GOC_SUCCESS);
          uint64_t digest = goc_test::capture_hash_seed;
          hash_words(digest, words[3]);
          uint64_t upper_digest = goc_test::capture_hash_seed;
          hash_words(upper_digest, words[3] + 32);
          EXPECT_EQ(upper_digest, digest);
          EXPECT_EQ(digest, goc_test::interp32_full_digests[(op * 16 + m) * 8 + wait])
              << cpu << "/" << op << "/" << m << "/" << wait;
        }
}

TEST(Interp32, Wave64HardwareReadsInactiveSourceLanes) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned mi = 0; mi < 36; ++mi) {
      const uint32_t first[] = {0, UINT32_MAX, 0x55555555, 0xaaaaaaaa};
      uint64_t exec_mask = uint64_t(mi < 4 ? first[mi] : 1U << (mi - 4)) << 32;
      uint64_t digest = goc_test::capture_hash_seed;
      for (unsigned op = 0; op < 2; ++op)
        for (unsigned m = 0; m < 16; ++m) {
          uint32_t words[4][64];
          wave64_capture_inputs(words);
          const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
          uint32_t *d[] = {words[3]};
          ASSERT_EQ(wave64_functions[op](cpu, exec_mask, goc_test::interp32_mode(m), d, a, b, c),
                    GOC_SUCCESS);
          hash_words(digest, words[3] + 32);
          for (unsigned lane = 0; lane < 32; ++lane)
            EXPECT_EQ(words[3][lane], 0xdeadbeefU);
        }
      EXPECT_EQ(digest, goc_test::interp32_masked_digests[mi]) << cpu << "/" << mi;
    }
}

TEST(Interp32, Wave64MasksModifiersAliasesAndUnalignedStorage) {
  const unsigned sources[][3] = {{0, 1, 2}, {0, 0, 2}, {0, 1, 0}, {0, 1, 1}, {0, 0, 0}};
  for (unsigned op = 0; op < 2; ++op)
    for (unsigned m = 0; m < 16; ++m)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (const auto &source : sources)
          for (unsigned target = 0; target < 4; ++target)
            for (uint64_t exec_mask : {uint64_t(UINT64_MAX), uint64_t(0xffffffff00000000ULL),
                                       uint64_t(0x87654321fedcba98ULL)}) {
              uint32_t words[4][67], expected[4][67];
              for (unsigned reg = 0; reg < 4; ++reg)
                for (unsigned lane = 0; lane < 67; ++lane)
                  words[reg][lane] = expected[reg][lane] =
                      goc_test::interp32_bits(float(int(lane * 3 + reg * 17) - 45) * 0.125f);
              for (unsigned lane = 0; lane < 64; ++lane)
                if ((exec_mask >> lane) & 1)
                  expected[target][lane + 1] = goc_test::interp32_bits(
                      goc_test::interp32_reference(op, words[source[0]] + 1, words[source[1]] + 1,
                                                   words[source[2]] + 1, lane, m));
              const uint32_t *a[] = {words[source[0]] + 1}, *b[] = {words[source[1]] + 1},
                             *c[] = {words[source[2]] + 1};
              uint32_t *d[] = {words[target] + 1};
              ASSERT_EQ(wave64_functions[op](cpu, exec_mask, goc_test::interp32_mode(m, m % 8), d,
                                             a, b, c),
                        GOC_SUCCESS);
              for (unsigned reg = 0; reg < 4; ++reg)
                ASSERT_TRUE(std::equal(words[reg], words[reg] + 67, expected[reg]))
                    << cpu << "/" << op << "/" << m << "/" << target;
            }
}

TEST(Interp32, Wave64SpecialValuesAndRandomBits) {
  const uint32_t edges[] = {0,          0x80000000, 0x3f800000, 0xbf800000, 0x7f800000, 0xff800000,
                            0x7fc12345, 0x7f812345, 1,          0x80000001, 0x7f7fffff, 0xff7fffff,
                            0x00800000, 0x80800000, 0x3f000000, 0xbf000000};
  for (unsigned op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned m = 0; m < 16; ++m) {
        std::mt19937 random(971);
        for (unsigned batch = 0; batch < 96; ++batch) {
          uint32_t words[4][64];
          for (unsigned reg = 0; reg < 3; ++reg)
            for (unsigned lane = 0; lane < 64; ++lane)
              words[reg][lane] = batch < 32 ? edges[(batch + lane * (reg + 1) + reg * 7) % 16]
                                            : uint32_t(random());
          const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
          uint32_t *d[] = {words[3]};
          ASSERT_EQ(wave64_functions[op](cpu, UINT64_MAX, goc_test::interp32_mode(m), d, a, b, c),
                    GOC_SUCCESS);
          for (unsigned lane = 0; lane < 64; ++lane)
            ASSERT_TRUE(equal_float(words[3][lane], goc_test::interp32_reference(
                                                        op, words[0], words[1], words[2], lane, m)))
                << cpu << "/" << op << "/" << m << "/" << batch << "/" << lane;
        }
      }
}

TEST(Interp32, Wave64ValidationAndEmptyExec) {
  uint32_t known = goc_test::interp32_mode(15, 7);
  for (auto fn : wave64_functions) {
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

TEST(Interp32, SharedArchitectureContracts) {
  const Fn narrow[] = {goc_v_interp_p10_f32,         goc_v_interp_p2_f32,
                       goc_v_interp_p10_f16_f32,     goc_v_interp_p2_f16_f32,
                       goc_v_interp_p10_rtz_f16_f32, goc_v_interp_p2_rtz_f16_f32};
  const decltype(&goc_v_interp_p10_f32_wave64) wide[] = {
      goc_v_interp_p10_f32_wave64,         goc_v_interp_p2_f32_wave64,
      goc_v_interp_p10_f16_f32_wave64,     goc_v_interp_p2_f16_f32_wave64,
      goc_v_interp_p10_rtz_f16_f32_wave64, goc_v_interp_p2_rtz_f16_f32_wave64};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned op = 0; op < 6; ++op)
      for (unsigned width : {32U, 64U})
        for (uint64_t exec_mask : {0ULL, 0xa55aa55a5aa55aa5ULL, ~0ULL}) {
          SCOPED_TRACE(::testing::Message()
                       << cpu << '/' << op << '/' << width << '/' << exec_mask);
          uint32_t a[64], b[64], c[64], d[64];
          std::fill(a, a + 64, op < 2 ? 0x3f800000U : 0x3c003c00U);
          std::fill(b, b + 64, 0x3f800000U);
          std::fill(c, c + 64, op < 2 || (op & 1) ? 0x3f800000U : 0x3c003c00U);
          const uint32_t *pa = a, *pb = b, *pc = c;
          uint32_t *pd = d;
          auto call = [&](uint64_t flags) {
            return width == 32 ? narrow[op](flags, uint32_t(exec_mask), 0, &pd, &pa, &pb, &pc)
                               : wide[op](flags, exec_mask, 0, &pd, &pa, &pb, &pc);
          };
          std::fill(d, d + 64, 0xdeadbeefU);
          // Upstream interpolation handlers are stubs, not an exact model.
          EXPECT_EQ(call(cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT),
                    GOC_ERROR_UNSUPPORTED_SEMANTICS);
          for (uint64_t guest :
               {GOC_FP_IEEE_MODE, GOC_FP_FLUSH_INPUT_DENORMALS, GOC_FP_FLUSH_OUTPUT_DENORMALS})
            EXPECT_EQ(call(cpu | guest), GOC_ERROR_INVALID_FLAGS);
          for (auto word : d)
            EXPECT_EQ(word, 0xdeadbeefU);
          for (uint64_t semantics : {uint64_t(0), GOC_SEMANTICS_EXACT_EMPIRICAL}) {
            std::fill(d, d + 64, 0xdeadbeefU);
            EXPECT_EQ(call(cpu | semantics), GOC_SUCCESS);
            for (unsigned lane = 0; lane < 64; ++lane) {
              uint32_t result = op >= 2 && (op & 1) ? 0xdead4000U : 0x40000000U;
              EXPECT_EQ(d[lane], lane < width && ((exec_mask >> lane) & 1) ? result : 0xdeadbeefU);
            }
          }
        }
}

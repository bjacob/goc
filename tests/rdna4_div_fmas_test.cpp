// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "rdna4_div_fmas_hardware.h"
#include "rdna4_exec_masks.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_rdna4_v_div_fmas_f32);
const Fn functions[] = {goc_rdna4_v_div_fmas_f32, goc_rdna4_v_div_fmas_f64};
const uint64_t exact = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;

} // namespace

TEST(DivFmas, HardwareCartesianAndRandomCorpora) {
  for (unsigned op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned corpus = 0; corpus < 2; ++corpus)
        for (unsigned col = 0; col < 16; ++col) {
          uint32_t state = 0x9174ab23;
          uint64_t digest = goc_test::capture_hash_seed;
          for (unsigned start = 0; start < 4096; start += 32) {
            uint32_t data[8][32] = {};
            const uint32_t *a[] = {data[0], data[1]}, *b[] = {data[2], data[3]},
                           *c[] = {data[4], data[5]};
            uint32_t *d[] = {data[6], data[7]};
            for (unsigned lane = 0; lane < 32; ++lane) {
              unsigned index = start + lane,
                       indices[] = {index / 256, (index / 16) % 16, index % 16};
              uint64_t random_values[3];
              if (corpus)
                goc_test::fmas_capture_random(op, index, state, random_values);
              for (unsigned operand = 0; operand < 3; ++operand) {
                uint64_t value = corpus ? random_values[operand]
                                        : goc_test::fmas_capture_values[op][indices[operand]];
                data[operand * 2][lane] = uint32_t(value);
                data[operand * 2 + 1][lane] = uint32_t(value >> 32);
              }
            }
            ASSERT_EQ(functions[op](cpu | exact, UINT32_MAX, goc_test::fmas_capture_modes[col % 8],
                                    d, a, b, c, col / 8 ? UINT32_MAX : 0),
                      GOC_SUCCESS);
            for (unsigned lane = 0; lane < 32; ++lane) {
              uint64_t value = data[6][lane];
              if (op)
                value |= uint64_t(data[7][lane]) << 32;
              digest = goc_test::capture_hash_bytes(digest, value, (op ? 8u : 4u));
            }
          }
          ASSERT_EQ(digest, (corpus ? goc_test::fmas_random_digests[op][col]
                                    : goc_test::fmas_capture_digests[op][col]))
              << op << "/" << cpu << "/" << col;
        }
}

TEST(DivFmas, AllModifiersRandomBitsAndConditionMasks) {
  std::mt19937 random(819542);
  for (unsigned op = 0; op < 2; ++op)
    for (uint32_t mode = 0; mode < 512; ++mode)
      for (unsigned sample = 0; sample < 8; ++sample) {
        uint32_t data[8][32], expected[2][32], condition = random();
        for (auto &reg : data)
          for (auto &word : reg)
            word = random();
        // Include special values and exact cancellation alongside unrestricted bits.
        for (unsigned lane = 0; lane < 16; ++lane)
          for (unsigned operand = 0; operand < 3; ++operand) {
            uint64_t value =
                goc_test::fmas_capture_values[op][(lane * (operand * 2 + 1) + sample) % 16];
            data[operand * 2][lane] = uint32_t(value);
            data[operand * 2 + 1][lane] = uint32_t(value >> 32);
          }
        const uint32_t *a[] = {data[0], data[1]}, *b[] = {data[2], data[3]},
                       *c[] = {data[4], data[5]};
        uint32_t *d[] = {data[6], data[7]}, *gold[] = {expected[0], expected[1]};
        ASSERT_EQ(functions[op](exact, UINT32_MAX, mode, gold, a, b, c, condition), GOC_SUCCESS);
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
          ASSERT_EQ(
              functions[op](cpu | exact | GOC_FP16_OVFL, UINT32_MAX, mode, d, a, b, c, condition),
              GOC_SUCCESS);
          for (unsigned reg = 0; reg < (op ? 2u : 1u); ++reg)
            for (unsigned lane = 0; lane < 32; ++lane)
              ASSERT_EQ(d[reg][lane], expected[reg][lane])
                  << op << "/" << cpu << "/" << mode << "/" << lane;
        }
      }
}

TEST(DivFmas, MasksAndCrossRegisterAliases) {
  std::mt19937 random(975283);
  for (unsigned op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < 8; ++variant)
        for (uint32_t mask : rdna4_exec_masks())
          for (unsigned first = 0; first < 8; first += (op ? 1 : 2))
            for (unsigned second = 0; second < (op ? 8u : 1u); ++second) {
              uint32_t data[8][34], expected[8][34], result[2][32], condition = random();
              for (auto &reg : data)
                for (auto &word : reg)
                  word = random();
              std::memcpy(expected, data, sizeof(data));
              uint32_t mode = goc_test::fmas_capture_modes[variant];
              const uint32_t *a[] = {data[0] + 1, data[1] + 1}, *b[] = {data[2] + 1, data[3] + 1},
                             *c[] = {data[4] + 1, data[5] + 1};
              uint32_t *d[] = {data[first] + 1, data[second] + 1}, *gold[] = {result[0], result[1]};
              ASSERT_EQ(functions[op](exact, UINT32_MAX, mode, gold, a, b, c, condition),
                        GOC_SUCCESS);
              unsigned dest[] = {first, second};
              for (unsigned reg = 0; reg < (op ? 2u : 1u); ++reg)
                for (unsigned lane = 0; lane < 32; ++lane)
                  if ((mask >> lane) & 1)
                    expected[dest[reg]][lane + 1] = result[reg][lane];
              ASSERT_EQ(
                  functions[op](cpu | (variant & 1 ? exact : 0), mask, mode, d, a, b, c, condition),
                  GOC_SUCCESS);
              ASSERT_EQ(std::memcmp(data, expected, sizeof(data)), 0)
                  << op << "/" << cpu << "/" << variant << "/" << first << "/" << second;
            }
}

TEST(DivFmas, SharedSourcesAndHostEnvironment) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (unsigned op = 0; op < 2; ++op)
    for (uint32_t mode = 0; mode < 512; ++mode) {
      uint32_t data[4][32], expected[2][32];
      for (unsigned lane = 0; lane < 32; ++lane) {
        uint64_t value = goc_test::fmas_capture_values[op][lane % 16];
        data[0][lane] = uint32_t(value);
        data[1][lane] = uint32_t(value >> 32);
      }
      const uint32_t *a[] = {data[0], data[1]};
      uint32_t *d[] = {data[2], data[3]}, *gold[] = {expected[0], expected[1]};
      ASSERT_EQ(functions[op](exact, UINT32_MAX, mode, gold, a, a, a, 0xa5a5a5a5), GOC_SUCCESS);
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
          EXPECT_EQ(std::fesetround(rounding), 0);
          EXPECT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
          if (mode & 1) {
            EXPECT_EQ(std::feraiseexcept(FE_INVALID | FE_INEXACT), 0);
          }
          int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
          EXPECT_EQ(functions[op](cpu | exact, UINT32_MAX, mode, d, a, a, a, 0xa5a5a5a5),
                    GOC_SUCCESS);
          EXPECT_EQ(std::fegetround(), rounding);
          EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
          for (unsigned reg = 0; reg < (op ? 2u : 1u); ++reg)
            for (unsigned lane = 0; lane < 32; ++lane)
              ASSERT_EQ(d[reg][lane], expected[reg][lane]);
        }
    }
}

TEST(DivFmas, ValidationAndZeroExec) {
  for (auto fn : functions)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t output[32];
      std::fill_n(output, 32, 0xdeadbeef);
      uint32_t *d[] = {output, output};
      for (unsigned bit = 9; bit < 32; ++bit) {
        EXPECT_EQ(fn(cpu, UINT32_MAX, 1u << bit, d, nullptr, nullptr, nullptr, UINT32_MAX),
                  GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(fn(cpu, 0, 1u << bit, nullptr, nullptr, nullptr, nullptr, UINT32_MAX),
                  GOC_ERROR_INVALID_FLAGS);
      }
      EXPECT_EQ(fn(cpu | (1ull << 63), UINT32_MAX, 0, d, nullptr, nullptr, nullptr, 0),
                GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(cpu | 2 * GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, UINT32_MAX, 0, d,
                   nullptr, nullptr, nullptr, 0),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
      for (auto word : output)
        EXPECT_EQ(word, 0xdeadbeef);
      EXPECT_EQ(fn(cpu | exact, 0U, 0, nullptr, nullptr, nullptr, nullptr, UINT32_MAX),
                GOC_SUCCESS);
      EXPECT_EQ(
          fn(cpu | 2 * GOC_SEMANTICS_EXACT_EMPIRICAL, 0, 0, nullptr, nullptr, nullptr, nullptr, 0),
          GOC_SUCCESS);
    }
}

TEST(DivFmas, FusedRoundingAndOutputModifierBoundaries) {
  for (unsigned op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned col = 0; col < 16; ++col) {
        uint32_t data[8][32];
        const uint32_t *a[] = {data[0], data[1]}, *b[] = {data[2], data[3]},
                       *c[] = {data[4], data[5]};
        uint32_t *d[] = {data[6], data[7]};
        for (unsigned lane = 0; lane < 32; ++lane)
          for (unsigned operand = 0; operand < 3; ++operand) {
            uint64_t value = goc_test::fmas_boundary[op][lane % 8][operand];
            data[operand * 2][lane] = uint32_t(value);
            data[operand * 2 + 1][lane] = uint32_t(value >> 32);
          }
        ASSERT_EQ(functions[op](cpu | exact, UINT32_MAX, goc_test::fmas_capture_modes[col % 8], d,
                                a, b, c, col / 8 ? UINT32_MAX : 0),
                  GOC_SUCCESS);
        for (unsigned lane = 0; lane < 32; ++lane) {
          uint64_t result = data[6][lane];
          if (op)
            result |= uint64_t(data[7][lane]) << 32;
          EXPECT_EQ(result, goc_test::fmas_boundary[op][lane % 8][col + 3])
              << op << "/" << cpu << "/" << col << "/" << lane;
        }
      }
}

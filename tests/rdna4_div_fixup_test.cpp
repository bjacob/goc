// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_div_fixup_hardware.h"
#include "rdna4_div_fixup_reference.h"
#include "rdna4_exec_masks.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_rdna4_v_div_fixup_f16);
const Fn functions[] = {goc_rdna4_v_div_fixup_f16, goc_rdna4_v_div_fixup_f32,
                        goc_rdna4_v_div_fixup_f64};
const unsigned widths[] = {16, 32, 64};
const uint64_t exact = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;

uint64_t load(unsigned width, const uint32_t *const *v, unsigned lane, uint32_t mode,
              unsigned operand) {
  uint64_t value = v[0][lane];
  if (width == 64)
    value |= uint64_t(v[1][lane]) << 32;
  if (width == 16)
    value = (value >> (mode & (GOC_ALU_HIGH_A << operand) ? 16 : 0)) & 65535;
  return value;
}

void store(unsigned width, uint32_t *const *d, unsigned lane, uint32_t mode, uint64_t value) {
  if (width == 16) {
    unsigned shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
    d[0][lane] = (d[0][lane] & ~(65535u << shift)) | (uint32_t(value) << shift);
  } else {
    d[0][lane] = uint32_t(value);
    if (width == 64)
      d[1][lane] = uint32_t(value >> 32);
  }
}

} // namespace

TEST(DivFixup, HardwareCartesianCorpus) {
  for (unsigned op = 0; op < 3; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned col = 0; col < 10; ++col) {
        uint64_t digest = UINT64_C(14695981039346656037);
        for (unsigned start = 0; start < 4096; start += 32) {
          uint32_t data[8][32] = {};
          const uint32_t *a[] = {data[0], data[1]}, *b[] = {data[2], data[3]},
                         *c[] = {data[4], data[5]};
          uint32_t *d[] = {data[6], data[7]};
          for (unsigned lane = 0; lane < 32; ++lane) {
            unsigned index = start + lane;
            unsigned indices[] = {index / 256, (index / 16) % 16, index % 16};
            for (unsigned operand = 0; operand < 3; ++operand) {
              uint64_t value = goc_test::fixup_capture_values[op][indices[operand]];
              data[operand * 2][lane] = uint32_t(value);
              data[operand * 2 + 1][lane] = uint32_t(value >> 32);
            }
            data[6][lane] = 0x12345678;
          }
          ASSERT_EQ(functions[op](cpu | exact | (col >= 5 ? GOC_FP16_OVFL : 0), UINT32_MAX,
                                  goc_test::fixup_capture_modes[col % 5], d, a, b, c),
                    GOC_SUCCESS);
          for (unsigned lane = 0; lane < 32; ++lane) {
            uint64_t word = data[6][lane];
            if (op == 2)
              word |= uint64_t(data[7][lane]) << 32;
            for (unsigned shift = 0; shift < (op == 2 ? 64u : 32u); shift += 8) {
              digest ^= (word >> shift) & 255;
              digest *= UINT64_C(1099511628211);
            }
          }
        }
        EXPECT_EQ(digest, goc_test::fixup_capture_digests[op][col])
            << op << "/" << cpu << "/" << col;
      }
}

TEST(DivFixup, EveryModifierHalfSelectorAndRandomBits) {
  std::mt19937 random(77892);
  for (unsigned op = 0; op < 3; ++op)
    for (uint32_t mode = 0; mode < (op == 0 ? 8192u : 512u); ++mode)
      for (bool sat : {false, true}) {
        uint32_t data[8][32], expected[2][32];
        for (auto &reg : data)
          for (auto &word : reg)
            word = random();
        const uint32_t *a[] = {data[0], data[1]}, *b[] = {data[2], data[3]},
                       *c[] = {data[4], data[5]};
        uint32_t *d[] = {data[6], data[7]}, *gold[] = {expected[0], expected[1]};
        for (unsigned lane = 0; lane < 16; ++lane)
          for (unsigned operand = 0; operand < 3; ++operand) {
            uint64_t value =
                goc_test::fixup_capture_values[op]
                                              [(lane * (operand * 2 + 1) + mode / 64 + operand) %
                                               16];
            if (op == 0) {
              unsigned shift = mode & (GOC_ALU_HIGH_A << operand) ? 16 : 0;
              data[2 * operand][lane] =
                  (data[2 * operand][lane] & ~(65535u << shift)) | (uint32_t(value) << shift);
            } else {
              data[2 * operand][lane] = uint32_t(value);
              data[2 * operand + 1][lane] = uint32_t(value >> 32);
            }
          }
        std::memcpy(expected[0], data[6], sizeof(expected[0]));
        std::memcpy(expected[1], data[7], sizeof(expected[1]));
        for (unsigned lane = 0; lane < 32; ++lane)
          store(widths[op], gold, lane, mode,
                goc_test::fixup_reference(widths[op], load(widths[op], a, lane, mode, 0),
                                          load(widths[op], b, lane, mode, 1),
                                          load(widths[op], c, lane, mode, 2), mode, sat));
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
          ASSERT_EQ(
              functions[op](cpu | exact | (sat ? GOC_FP16_OVFL : 0), UINT32_MAX, mode, d, a, b, c),
              GOC_SUCCESS);
          for (unsigned reg = 0; reg < (op == 2 ? 2u : 1u); ++reg)
            for (unsigned lane = 0; lane < 32; ++lane)
              ASSERT_EQ(d[reg][lane], expected[reg][lane])
                  << op << "/" << mode << "/" << sat << "/" << cpu << "/" << lane;
        }
      }
}

TEST(DivFixup, MasksAndCrossRegisterAliases) {
  std::mt19937 random(687531);
  for (unsigned op = 0; op < 3; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < 8; ++variant) {
        uint32_t mode =
            (variant & 1 ? 63 : 0) | ((variant >> 1) << 6) | (variant & 1 ? GOC_ALU_CLAMP : 0);
        if (op == 0)
          mode |= (variant & 7) << 9 | (variant & 1 ? GOC_ALU_HIGH_D : 0);
        for (uint64_t mask : rdna4_exec_masks())
          for (unsigned first = 0; first < 8; first += (op == 2 ? 1 : 2))
            for (unsigned second = 0; second < (op == 2 ? 8u : 1u); ++second) {
              uint32_t data[8][34], expected[8][34];
              for (unsigned reg = 0; reg < 8; ++reg)
                for (unsigned word = 0; word < 34; ++word)
                  data[reg][word] = expected[reg][word] = random();
              const uint32_t *a[] = {data[0] + 1, data[1] + 1}, *b[] = {data[2] + 1, data[3] + 1},
                             *c[] = {data[4] + 1, data[5] + 1};
              uint32_t *d[] = {data[first] + 1, data[second] + 1},
                       *gold[] = {expected[first] + 1, expected[second] + 1};
              for (unsigned lane = 0; lane < 32; ++lane)
                if ((mask >> lane) & 1)
                  store(widths[op], gold, lane, mode,
                        goc_test::fixup_reference(widths[op], load(widths[op], a, lane, mode, 0),
                                                  load(widths[op], b, lane, mode, 1),
                                                  load(widths[op], c, lane, mode, 2), mode,
                                                  variant & 1));
              ASSERT_EQ(functions[op](cpu | (variant & 1 ? GOC_FP16_OVFL | exact : 0), mask, mode,
                                      d, a, b, c),
                        GOC_SUCCESS);
              for (unsigned reg = 0; reg < 8; ++reg)
                for (unsigned word = 0; word < 34; ++word)
                  ASSERT_EQ(data[reg][word], expected[reg][word])
                      << op << "/" << cpu << "/" << mode << "/" << first << "/" << second;
            }
      }
}

TEST(DivFixup, SharedSourcesAndHostEnvironment) {
  fenv_t saved;
  ASSERT_EQ(std::fegetenv(&saved), 0);
  for (int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
    EXPECT_EQ(std::fesetround(rounding), 0);
    for (unsigned op = 0; op < 3; ++op)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (uint32_t mode = 0; mode < 512; ++mode) {
          uint32_t data[4][32], expected[2][32];
          for (unsigned lane = 0; lane < 32; ++lane) {
            uint64_t value = goc_test::fixup_capture_values[op][lane % 16];
            data[0][lane] = uint32_t(value);
            data[1][lane] = uint32_t(value >> 32);
            data[2][lane] = expected[0][lane] = 0x12345678;
            data[3][lane] = expected[1][lane] = 0xabcdef01;
          }
          const uint32_t *a[] = {data[0], data[1]};
          uint32_t *d[] = {data[2], data[3]}, *gold[] = {expected[0], expected[1]};
          for (unsigned lane = 0; lane < 32; ++lane) {
            uint64_t value = load(widths[op], a, lane, mode, 0);
            store(widths[op], gold, lane, mode,
                  goc_test::fixup_reference(widths[op], value, value, value, mode, true));
          }
          EXPECT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
          EXPECT_EQ(std::feraiseexcept(FE_INEXACT | FE_INVALID), 0);
          int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
          EXPECT_EQ(functions[op](cpu | exact | GOC_FP16_OVFL, UINT32_MAX, mode, d, a, a, a),
                    GOC_SUCCESS);
          EXPECT_EQ(std::fegetround(), rounding);
          EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
          for (unsigned reg = 0; reg < (op == 2 ? 2u : 1u); ++reg)
            for (unsigned lane = 0; lane < 32; ++lane)
              EXPECT_EQ(d[reg][lane], expected[reg][lane]);
        }
  }
  EXPECT_EQ(std::fesetenv(&saved), 0);
}

TEST(DivFixup, ValidationAndFallback) {
  for (unsigned op = 0; op < 3; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t input[32] = {}, output[32];
      std::fill_n(output, 32, 0xdeadbeef);
      const uint32_t *a[] = {input, input};
      uint32_t *d[] = {output, output};
      for (unsigned bit = op == 0 ? 13 : 9; bit < 32; ++bit) {
        EXPECT_EQ(functions[op](cpu, UINT32_MAX, 1u << bit, d, a, a, a), GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(functions[op](cpu, 0, 1u << bit, nullptr, nullptr, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
      }
      EXPECT_EQ(functions[op](cpu | (1ull << 63), UINT32_MAX, 0, d, a, a, a),
                GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(functions[op](cpu | 2 * GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                              UINT32_MAX, 0, d, a, a, a),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
      for (auto word : output)
        EXPECT_EQ(word, 0xdeadbeef);
      EXPECT_EQ(
          functions[op](cpu | exact, 0xffffffff00000000ull, 0, nullptr, nullptr, nullptr, nullptr),
          GOC_SUCCESS);
      EXPECT_EQ(functions[op](cpu, 0, 0, nullptr, nullptr, nullptr, nullptr), GOC_SUCCESS);
      EXPECT_EQ(functions[op](cpu | 2 * GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, 0, d, a, a, a),
                GOC_SUCCESS);
      for (auto word : output)
        EXPECT_EQ(word, op == 0 ? 0xdeadfe00 : op == 1 ? 0xffc00000 : 0xfff80000);
    }
}

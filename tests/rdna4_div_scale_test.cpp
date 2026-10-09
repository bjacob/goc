// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_div_scale_hardware.h"
#include "rdna4_exec_masks.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_rdna4_v_div_scale_f32);
const Fn functions[] = {goc_rdna4_v_div_scale_f32, goc_rdna4_v_div_scale_f64};
const uint64_t exact = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;

uint32_t mode_for(unsigned variant) {
  return (variant & 7) | (((variant >> 3) & 3) << 6) | (variant & 32 ? GOC_ALU_CLAMP : 0);
}

void inputs(unsigned op, unsigned role, uint32_t mode, uint64_t b, uint64_t c, uint32_t data[8][32],
            unsigned lane) {
  uint64_t a = role ? c : b;
  if (bool(mode & GOC_ALU_NEG_A) != bool(mode & (role ? GOC_ALU_NEG_C : GOC_ALU_NEG_B)))
    a ^= UINT64_C(1) << (op ? 63 : 31);
  uint64_t values[] = {a, b, c};
  for (unsigned i = 0; i < 3; ++i) {
    data[2 * i][lane] = uint32_t(values[i]);
    data[2 * i + 1][lane] = uint32_t(values[i] >> 32);
  }
}

} // namespace

TEST(DivScale, HardwareCartesianCorpus) {
  for (unsigned op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned col = 0; col < 128; ++col) {
        uint64_t digest = UINT64_C(14695981039346656037);
        for (unsigned start = 0; start < 576; start += 32) {
          uint32_t data[8][32] = {}, condition = 0;
          const uint32_t *a[] = {data[0], data[1]}, *b[] = {data[2], data[3]},
                         *c[] = {data[4], data[5]};
          uint32_t *d[] = {data[6], data[7]};
          for (unsigned lane = 0; lane < 32; ++lane) {
            unsigned index = start + lane;
            inputs(op, col / 64, mode_for(col % 64), goc_test::scale_capture_values[op][index / 24],
                   goc_test::scale_capture_values[op][index % 24], data, lane);
          }
          ASSERT_EQ(
              functions[op](cpu | exact, UINT32_MAX, mode_for(col % 64), d, &condition, a, b, c),
              GOC_SUCCESS);
          for (unsigned lane = 0; lane < 32; ++lane) {
            uint64_t value = data[6][lane];
            if (op)
              value |= uint64_t(data[7][lane]) << 32;
            for (unsigned byte = 0; byte < (op ? 8u : 4u); ++byte) {
              digest ^= (value >> (byte * 8)) & 255;
              digest *= UINT64_C(1099511628211);
            }
            digest ^= (condition >> lane) & 1;
            digest *= UINT64_C(1099511628211);
          }
        }
        ASSERT_EQ(digest, goc_test::scale_capture_digests[op][col])
            << op << "/" << cpu << "/" << col;
      }
}

TEST(DivScale, RandomBitsAllModifiersAndHostEnvironment) {
  std::mt19937 random(569827);
  fenv_t saved;
  ASSERT_EQ(std::fegetenv(&saved), 0);
  for (unsigned op = 0; op < 2; ++op)
    for (unsigned variant = 0; variant < 64; ++variant)
      for (unsigned sample = 0; sample < 16; ++sample) {
        uint32_t data[8][32] = {}, expected[2][32], expected_condition;
        const uint32_t *a[] = {data[0], data[1]}, *b[] = {data[2], data[3]},
                       *c[] = {data[4], data[5]};
        uint32_t *d[] = {data[6], data[7]};
        for (unsigned lane = 0; lane < 32; ++lane) {
          uint64_t bv = random(), cv = random();
          if (op) {
            bv |= uint64_t(random()) << 32;
            cv |= uint64_t(random()) << 32;
          }
          inputs(op, lane & 1, mode_for(variant), bv, cv, data, lane);
        }
        ASSERT_EQ(
            functions[op](exact, UINT32_MAX, mode_for(variant), d, &expected_condition, a, b, c),
            GOC_SUCCESS);
        std::memcpy(expected, data[6], sizeof(expected));
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
          for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
            EXPECT_EQ(std::fesetround(rounding), 0);
            EXPECT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
            if (sample & 1) {
              EXPECT_EQ(std::feraiseexcept(FE_INVALID | FE_INEXACT), 0);
            }
            int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
            uint32_t condition = 0;
            EXPECT_EQ(functions[op](cpu | exact | GOC_FP16_OVFL, UINT32_MAX, mode_for(variant), d,
                                    &condition, a, b, c),
                      GOC_SUCCESS);
            EXPECT_EQ(std::fegetround(), rounding);
            EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
            ASSERT_EQ(condition, expected_condition);
            for (unsigned reg = 0; reg < (op ? 2u : 1u); ++reg)
              for (unsigned lane = 0; lane < 32; ++lane)
                ASSERT_EQ(d[reg][lane], expected[reg][lane])
                    << op << "/" << cpu << "/" << variant << "/" << lane;
          }
      }
  EXPECT_EQ(std::fesetenv(&saved), 0);
}

TEST(DivScale, MasksAndCrossRegisterAliases) {
  std::mt19937 random(871923);
  for (unsigned op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < 64; ++variant)
        for (uint64_t mask : rdna4_exec_masks())
          for (unsigned first = 0; first < 8; first += (op ? 1 : 2))
            for (unsigned second = 0; second < (op ? 8u : 1u); ++second) {
              uint32_t data[8][32], expected[8][32], result[2][32], condition = 0xa5a5a5a5,
                                                                    expected_condition;
              for (auto &reg : data)
                for (auto &word : reg)
                  word = random();
              for (unsigned lane = 0; lane < 32; ++lane)
                inputs(op, lane & 1, mode_for(variant),
                       goc_test::scale_capture_values[op][(lane + variant) % 24],
                       goc_test::scale_capture_values[op][(lane * 7 + variant) % 24], data, lane);
              std::memcpy(expected, data, sizeof(data));
              const uint32_t *a[] = {data[0], data[1]}, *b[] = {data[2], data[3]},
                             *c[] = {data[4], data[5]};
              uint32_t *gold[] = {result[0], result[1]}, *d[] = {data[first], data[second]};
              ASSERT_EQ(functions[op](exact, UINT32_MAX, mode_for(variant), gold,
                                      &expected_condition, a, b, c),
                        GOC_SUCCESS);
              unsigned dest[] = {first, second};
              for (unsigned reg = 0; reg < (op ? 2u : 1u); ++reg)
                for (unsigned lane = 0; lane < 32; ++lane)
                  if ((mask >> lane) & 1)
                    expected[dest[reg]][lane] = result[reg][lane];
              ASSERT_EQ(functions[op](cpu | (variant & 1 ? exact : 0), mask, mode_for(variant), d,
                                      &condition, a, b, c),
                        GOC_SUCCESS);
              ASSERT_EQ(condition, expected_condition & uint32_t(mask));
              ASSERT_EQ(std::memcmp(data, expected, sizeof(data)), 0)
                  << op << "/" << cpu << "/" << variant << "/" << first << "/" << second;
            }
}

TEST(DivScale, ValidationAndZeroExec) {
  for (auto fn : functions)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t condition = 0xdeadbeef;
      for (unsigned bit = 0; bit < 32; ++bit)
        if (!((1u << bit) & 0x1c7)) {
          EXPECT_EQ(fn(cpu, 0, 1u << bit, nullptr, &condition, nullptr, nullptr, nullptr),
                    GOC_ERROR_INVALID_FLAGS);
          EXPECT_EQ(condition, 0xdeadbeef);
        }
      EXPECT_EQ(fn(cpu | (1ull << 63), 0, 0, nullptr, &condition, nullptr, nullptr, nullptr),
                GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(cpu | 2 * GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0, nullptr,
                   &condition, nullptr, nullptr, nullptr),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
      EXPECT_EQ(condition, 0xdeadbeef);
      EXPECT_EQ(
          fn(cpu | exact, 0xffffffff00000000ull, 0, nullptr, &condition, nullptr, nullptr, nullptr),
          GOC_SUCCESS);
      EXPECT_EQ(condition, 0u);
      condition = 0xdeadbeef;
      EXPECT_EQ(fn(cpu | 2 * GOC_SEMANTICS_EXACT_EMPIRICAL, 0, 0, nullptr, &condition, nullptr,
                   nullptr, nullptr),
                GOC_SUCCESS);
      EXPECT_EQ(condition, 0u);
    }
}

TEST(DivScale, SharedSourcesUnalignedStorageAndScalarOutputAlias) {
  for (unsigned op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < 8; ++variant) {
        uint32_t source[2][34], output[2][34], expected[2][32], condition;
        for (unsigned lane = 0; lane < 34; ++lane) {
          uint64_t value = goc_test::scale_capture_values[op][lane % 24];
          source[0][lane] = uint32_t(value);
          source[1][lane] = uint32_t(value >> 32);
          output[0][lane] = output[1][lane] = 0xdeadbeef;
        }
        const uint32_t *a[] = {source[0] + 1, source[1] + 1};
        uint32_t *d[] = {output[0] + 1, output[1] + 1}, *gold[] = {expected[0], expected[1]};
        uint32_t mode = (variant & 1 ? 7 : 0) | ((variant >> 1) << 6);
        ASSERT_EQ(functions[op](exact, UINT32_MAX, mode, gold, &condition, a, a, a), GOC_SUCCESS);
        // A scalar destination may share storage; it is committed after all VGPR writes.
        ASSERT_EQ(functions[op](cpu | exact, UINT32_MAX, mode, d, d[0] + 13, a, a, a), GOC_SUCCESS);
        expected[0][13] = condition;
        for (unsigned reg = 0; reg < (op ? 2u : 1u); ++reg) {
          EXPECT_EQ(output[reg][0], 0xdeadbeef);
          EXPECT_EQ(output[reg][33], 0xdeadbeef);
          for (unsigned lane = 0; lane < 32; ++lane)
            EXPECT_EQ(d[reg][lane], expected[reg][lane]);
        }
      }
}

// SPDX-License-Identifier: MIT

#include "fp_environment.h"
#include "goc/goc.h"
#include "rdna4_exec_masks.h"
#include "rdna4_mad64_hardware.h"
#include "rdna4_mad64_reference.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_rdna4_v_mad_co_u64_u32);
const Fn functions[] = {goc_rdna4_v_mad_co_u64_u32, goc_rdna4_v_mad_co_i64_i32};
const uint64_t exact = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;

void digest_word(uint64_t &digest, uint64_t word, unsigned bytes) {
  for (unsigned byte = 0; byte < bytes; ++byte) {
    digest ^= (word >> (8 * byte)) & 255;
    digest *= UINT64_C(1099511628211);
  }
}

} // namespace

TEST(Mad64, HardwareCartesianCorpus) {
  const uint32_t masks[] = {UINT32_MAX, 0x33333333, 0};
  for (unsigned op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned clamp = 0; clamp < 2; ++clamp)
        for (unsigned m = 0; m < 3; ++m) {
          uint64_t digest = UINT64_C(14695981039346656037);
          for (unsigned start = 0; start < 4096; start += 32) {
            uint32_t data[6][32], carry = 0xa5a5a5a5;
            for (unsigned lane = 0; lane < 32; ++lane) {
              unsigned index = start + lane;
              data[0][lane] = goc_test::mad64_capture_factors[index / 256];
              data[1][lane] = goc_test::mad64_capture_factors[(index / 16) % 16];
              uint64_t c = goc_test::mad64_capture_addends[index % 16];
              data[2][lane] = uint32_t(c);
              data[3][lane] = uint32_t(c >> 32);
              data[4][lane] = 0xcafebabe;
              data[5][lane] = 0xdeadbeef;
            }
            const uint32_t *a[] = {data[0]}, *b[] = {data[1]}, *c[] = {data[2], data[3]};
            uint32_t *d[] = {data[4], data[5]};
            ASSERT_EQ(
                functions[op](cpu | exact, masks[m], clamp ? GOC_ALU_CLAMP : 0, d, &carry, a, b, c),
                GOC_SUCCESS);
            for (unsigned lane = 0; lane < 32; ++lane)
              digest_word(digest, data[4][lane] | (uint64_t(data[5][lane]) << 32), 8);
            digest_word(digest, carry, 4);
          }
          ASSERT_EQ(digest, goc_test::mad64_capture_digests[op][clamp][m])
              << op << "/" << cpu << "/" << clamp << "/" << m;
        }
}

TEST(Mad64, SignedScalarOutputIsTheExtendedSign) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (bool clamp : {false, true}) {
      uint32_t data[6][32];
      const uint32_t av[] = {0, 0x80000000, 0x80000000, 1}, bv[] = {0, 0x80000000, 0x7fffffff, 1};
      const uint64_t cv[] = {UINT64_MAX, UINT64_MAX >> 1, UINT64_C(1) << 63, 0};
      const uint64_t wrapped[] = {UINT64_MAX, UINT64_C(0xbfffffffffffffff),
                                  UINT64_C(0x4000000080000000), 1};
      const uint64_t saturated[] = {UINT64_MAX, UINT64_MAX >> 1, UINT64_C(1) << 63, 1};
      for (unsigned lane = 0; lane < 32; ++lane) {
        data[0][lane] = av[lane % 4];
        data[1][lane] = bv[lane % 4];
        data[2][lane] = uint32_t(cv[lane % 4]);
        data[3][lane] = uint32_t(cv[lane % 4] >> 32);
      }
      const uint32_t *a[] = {data[0]}, *b[] = {data[1]}, *c[] = {data[2], data[3]};
      uint32_t *d[] = {data[4], data[5]}, carry;
      ASSERT_EQ(
          functions[1](cpu | exact, UINT32_MAX, clamp ? GOC_ALU_CLAMP : 0, d, &carry, a, b, c),
          GOC_SUCCESS);
      // Lane 0 is negative without overflow; lane 1 overflows positively but has
      // a clear scalar bit. Lane 2 overflows negatively and retains a set bit.
      EXPECT_EQ(carry, 0x55555555u);
      for (unsigned lane = 0; lane < 32; ++lane)
        EXPECT_EQ(data[4][lane] | (uint64_t(data[5][lane]) << 32),
                  (clamp ? saturated : wrapped)[lane % 4]);
    }
}

TEST(Mad64, IndependentWideReferenceAndRandomInputs) {
  std::mt19937 random(784193);
  for (unsigned op = 0; op < 2; ++op)
    for (bool clamp : {false, true})
      for (unsigned sample = 0; sample < 128; ++sample) {
        uint32_t data[6][32], expected[2][32], expected_carry = 0;
        for (auto &reg : data)
          for (auto &word : reg)
            word = random();
        for (unsigned lane = 0; lane < 32; ++lane) {
          uint64_t c = data[2][lane] | (uint64_t(data[3][lane]) << 32);
          auto gold = goc_test::mad64_reference(op, data[0][lane], data[1][lane], c, clamp);
          expected[0][lane] = uint32_t(gold.value);
          expected[1][lane] = uint32_t(gold.value >> 32);
          expected_carry |= uint32_t(gold.carry) << lane;
        }
        const uint32_t *a[] = {data[0]}, *b[] = {data[1]}, *c[] = {data[2], data[3]};
        uint32_t *d[] = {data[4], data[5]};
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
          uint32_t carry = 0;
          ASSERT_EQ(functions[op](cpu | exact | GOC_FP16_OVFL, UINT32_MAX,
                                  clamp ? GOC_ALU_CLAMP : 0, d, &carry, a, b, c),
                    GOC_SUCCESS);
          EXPECT_EQ(carry, expected_carry);
          for (unsigned reg = 0; reg < 2; ++reg)
            for (unsigned lane = 0; lane < 32; ++lane)
              ASSERT_EQ(d[reg][lane], expected[reg][lane])
                  << op << "/" << cpu << "/" << clamp << "/" << lane;
        }
      }
}

TEST(Mad64, MasksAndCrossRegisterAliases) {
  std::mt19937 random(126923);
  for (unsigned op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool clamp : {false, true})
        for (uint32_t mask : rdna4_exec_masks())
          for (unsigned first = 0; first < 6; ++first)
            for (unsigned second = 0; second < 6; ++second) {
              uint32_t data[6][34], expected[6][34], result[2][32], carry = 0xa5a5a5a5,
                                                                    expected_carry = 0;
              for (auto &reg : data)
                for (auto &word : reg)
                  word = random();
              std::memcpy(expected, data, sizeof(data));
              const uint32_t *a[] = {data[0] + 1}, *b[] = {data[1] + 1},
                             *c[] = {data[2] + 1, data[3] + 1};
              uint32_t *d[] = {data[first] + 1, data[second] + 1};
              for (unsigned lane = 0; lane < 32; ++lane) {
                auto gold = goc_test::mad64_reference(
                    op, a[0][lane], b[0][lane], c[0][lane] | (uint64_t(c[1][lane]) << 32), clamp);
                result[0][lane] = uint32_t(gold.value);
                result[1][lane] = uint32_t(gold.value >> 32);
                expected_carry |= uint32_t(gold.carry) << lane;
              }
              unsigned dest[] = {first, second};
              for (unsigned reg = 0; reg < 2; ++reg)
                for (unsigned lane = 0; lane < 32; ++lane)
                  if ((mask >> lane) & 1)
                    expected[dest[reg]][lane + 1] = result[reg][lane];
              ASSERT_EQ(functions[op](cpu | (clamp ? exact : 0), mask, clamp ? GOC_ALU_CLAMP : 0, d,
                                      &carry, a, b, c),
                        GOC_SUCCESS);
              ASSERT_EQ(carry, expected_carry & mask);
              ASSERT_EQ(std::memcmp(data, expected, sizeof(data)), 0)
                  << op << "/" << cpu << "/" << clamp << "/" << first << "/" << second;
            }
}

TEST(Mad64, SharedSourcesScalarOutputAliasAndHostEnvironment) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (unsigned op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool clamp : {false, true})
        for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
          uint32_t input[32], output[2][32], expected[2][32], carry = 0;
          for (unsigned lane = 0; lane < 32; ++lane) {
            input[lane] = goc_test::mad64_capture_factors[lane % 16];
            auto gold = goc_test::mad64_reference(
                op, input[lane], input[lane], input[lane] | (uint64_t(input[lane]) << 32), clamp);
            expected[0][lane] = uint32_t(gold.value);
            expected[1][lane] = uint32_t(gold.value >> 32);
            carry |= uint32_t(gold.carry) << lane;
          }
          const uint32_t *a[] = {input, input};
          uint32_t *d[] = {output[0], output[1]};
          EXPECT_EQ(std::fesetround(rounding), 0);
          EXPECT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
          EXPECT_EQ(std::feraiseexcept(FE_INVALID | FE_INEXACT), 0);
          int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
          ASSERT_EQ(functions[op](cpu | exact, UINT32_MAX, clamp ? GOC_ALU_CLAMP : 0, d,
                                  output[1] + 13, a, a, a),
                    GOC_SUCCESS);
          expected[1][13] = carry;
          for (unsigned reg = 0; reg < 2; ++reg)
            for (unsigned lane = 0; lane < 32; ++lane)
              EXPECT_EQ(output[reg][lane], expected[reg][lane]);
          EXPECT_EQ(std::fegetround(), rounding);
          EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
        }
}

TEST(Mad64, ValidationAndZeroExec) {
  for (auto fn : functions)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t output[32], carry = 0x12345678;
      std::fill_n(output, 32, 0xdeadbeef);
      uint32_t *d[] = {output, output};
      for (unsigned bit = 0; bit < 32; ++bit)
        if ((1u << bit) != GOC_ALU_CLAMP) {
          EXPECT_EQ(fn(cpu, UINT32_MAX, 1u << bit, d, &carry, nullptr, nullptr, nullptr),
                    GOC_ERROR_INVALID_FLAGS);
          EXPECT_EQ(fn(cpu, 0, 1u << bit, nullptr, &carry, nullptr, nullptr, nullptr),
                    GOC_ERROR_INVALID_FLAGS);
        }
      EXPECT_EQ(fn(cpu | (1ull << 63), UINT32_MAX, 0, d, &carry, nullptr, nullptr, nullptr),
                GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(cpu | 2 * GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, UINT32_MAX, 0, d,
                   &carry, nullptr, nullptr, nullptr),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
      EXPECT_EQ(carry, 0x12345678);
      for (auto word : output)
        EXPECT_EQ(word, 0xdeadbeef);
      EXPECT_EQ(fn(cpu | exact, UINT32_C(0), 0, nullptr, &carry, nullptr, nullptr, nullptr),
                GOC_SUCCESS);
      EXPECT_EQ(carry, 0u);
      carry = 0x12345678;
      EXPECT_EQ(fn(cpu | 2 * GOC_SEMANTICS_EXACT_EMPIRICAL, 0, 0, nullptr, &carry, nullptr, nullptr,
                   nullptr),
                GOC_SUCCESS);
      EXPECT_EQ(carry, 0u);
    }
}

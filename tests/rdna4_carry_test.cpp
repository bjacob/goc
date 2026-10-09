// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "rdna4_carry_hardware.h"
#include "rdna4_carry_reference.h"
#include "rdna4_dpp_reference.h"
#include "rdna4_exec_masks.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {

const uint64_t exact = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;

} // namespace

TEST(Carry, HardwareCaptures) {
  const uint32_t masks[] = {UINT32_MAX, 0x33333333, 0};
  for (unsigned op = 0; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned clamp = 0; clamp < 2; ++clamp)
        for (unsigned m = 0; m < 3; ++m) {
          uint32_t data[3][32], carry = 0x96969696;
          for (unsigned lane = 0; lane < 32; ++lane) {
            data[0][lane] = goc_test::carry_capture_inputs[0][lane % 8];
            data[1][lane] = goc_test::carry_capture_inputs[1][lane % 8];
            data[2][lane] = 0xdeadbeef;
          }
          const uint32_t *a[] = {data[0]}, *b[] = {data[1]};
          uint32_t *d[] = {data[2]};
          ASSERT_EQ(goc_test::carry_functions[op](cpu | exact, masks[m], clamp ? GOC_ALU_CLAMP : 0,
                                                  d, &carry, a, b, 0xa5a5a5a5),
                    GOC_SUCCESS);
          EXPECT_EQ(carry, goc_test::carry_capture_masks[op][m]);
          for (unsigned lane = 0; lane < 32; ++lane)
            EXPECT_EQ(data[2][lane], ((masks[m] >> lane) & 1)
                                         ? goc_test::carry_capture_values[op][clamp][lane % 8]
                                         : 0xdeadbeef);
        }
}

TEST(Carry, BoundaryCartesianAndRandomInputs) {
  const uint32_t values[] = {0, 1, 2, 0x7ffffffe, 0x7fffffff, 0x80000000, 0xfffffffe, UINT32_MAX};
  std::mt19937 random(98938);
  for (unsigned op = 0; op < 6; ++op)
    for (bool clamp : {false, true})
      for (unsigned sample = 0; sample < 128; ++sample) {
        uint32_t data[3][32], expected[32],
            expected_carry = 0, ci = sample < 4 ? (sample & 2 ? UINT32_MAX : 0) : random();
        for (unsigned lane = 0; lane < 32; ++lane) {
          unsigned index = sample * 32 + lane;
          data[0][lane] = sample < 4 ? values[(index / 8) % 8] : random();
          data[1][lane] = sample < 4 ? values[index % 8] : random();
          auto gold =
              goc_test::carry_reference(op, data[0][lane], data[1][lane], (ci >> lane) & 1, clamp);
          expected[lane] = gold.value;
          expected_carry |= uint32_t(gold.carry) << lane;
        }
        const uint32_t *a[] = {data[0]}, *b[] = {data[1]};
        uint32_t *d[] = {data[2]};
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
          uint32_t carry = 0;
          ASSERT_EQ(goc_test::carry_functions[op](cpu | exact | GOC_FP16_OVFL, UINT32_MAX,
                                                  clamp ? GOC_ALU_CLAMP : 0, d, &carry, a, b, ci),
                    GOC_SUCCESS);
          EXPECT_EQ(carry, expected_carry);
          for (unsigned lane = 0; lane < 32; ++lane)
            ASSERT_EQ(d[0][lane], expected[lane])
                << op << "/" << cpu << "/" << clamp << "/" << lane;
        }
      }
}

TEST(Carry, ExecAndInputMasksUnalignedAliases) {
  std::mt19937 random(524578);
  auto masks = rdna4_exec_masks();
  for (unsigned op = 0; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool clamp : {false, true})
        for (uint32_t exec_mask : masks)
          for (uint64_t input_mask : masks)
            for (unsigned alias = 0; alias < 3; ++alias) {
              uint32_t data[3][34], expected[3][34], carry = 0xdeadbeef, wanted_carry = 0;
              for (auto &reg : data)
                for (auto &word : reg)
                  word = random();
              std::memcpy(expected, data, sizeof(data));
              const uint32_t *a[] = {data[0] + 1}, *b[] = {data[1] + 1};
              uint32_t *d[] = {data[alias] + 1};
              for (unsigned lane = 0; lane < 32; ++lane) {
                auto gold = goc_test::carry_reference(op, a[0][lane], b[0][lane],
                                                      (input_mask >> lane) & 1, clamp);
                if ((exec_mask >> lane) & 1) {
                  expected[alias][lane + 1] = gold.value;
                  wanted_carry |= uint32_t(gold.carry) << lane;
                }
              }
              ASSERT_EQ(goc_test::carry_functions[op](cpu | (clamp ? exact : 0), exec_mask,
                                                      clamp ? GOC_ALU_CLAMP : 0, d, &carry, a, b,
                                                      uint32_t(input_mask)),
                        GOC_SUCCESS);
              ASSERT_EQ(carry, wanted_carry);
              ASSERT_EQ(std::memcmp(data, expected, sizeof(data)), 0)
                  << op << "/" << cpu << "/" << clamp << "/" << alias;
            }
}

TEST(Carry, SharedSourcesScalarOutputAliasAndHostEnvironment) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (unsigned op = 0; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool clamp : {false, true})
        for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
          uint32_t data[32], expected[32], carry = 0;
          for (unsigned lane = 0; lane < 32; ++lane) {
            data[lane] = goc_test::carry_capture_inputs[0][lane % 8];
            auto gold = goc_test::carry_reference(op, data[lane], data[lane],
                                                  (0xa5a5a5a5 >> lane) & 1, clamp);
            expected[lane] = gold.value;
            carry |= uint32_t(gold.carry) << lane;
          }
          const uint32_t *a[] = {data};
          uint32_t *d[] = {data};
          EXPECT_EQ(std::fesetround(rounding), 0);
          EXPECT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
          EXPECT_EQ(std::feraiseexcept(FE_INVALID | FE_INEXACT), 0);
          int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
          ASSERT_EQ(goc_test::carry_functions[op](cpu | exact, UINT32_MAX,
                                                  clamp ? GOC_ALU_CLAMP : 0, d, data + 13, a, a,
                                                  0xa5a5a5a5),
                    GOC_SUCCESS);
          expected[13] = carry;
          for (unsigned lane = 0; lane < 32; ++lane)
            EXPECT_EQ(data[lane], expected[lane]);
          EXPECT_EQ(std::fegetround(), rounding);
          EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
        }
}

TEST(Carry, ValidationAndZeroExec) {
  for (auto fn : goc_test::carry_functions)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t output[32], carry = 0x12345678;
      std::fill_n(output, 32, 0xdeadbeef);
      uint32_t *d[] = {output};
      for (unsigned bit = 0; bit < 32; ++bit)
        if ((1u << bit) != GOC_ALU_CLAMP) {
          EXPECT_EQ(fn(cpu, UINT32_MAX, 1u << bit, d, &carry, nullptr, nullptr, UINT32_MAX),
                    GOC_ERROR_INVALID_FLAGS);
          EXPECT_EQ(fn(cpu, 0, 1u << bit, nullptr, &carry, nullptr, nullptr, UINT32_MAX),
                    GOC_ERROR_INVALID_FLAGS);
        }
      EXPECT_EQ(fn(cpu | (1ull << 63), UINT32_MAX, 0, d, &carry, nullptr, nullptr, 0),
                GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(cpu | 2 * GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, UINT32_MAX, 0, d,
                   &carry, nullptr, nullptr, 0),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
      EXPECT_EQ(carry, 0x12345678);
      for (auto word : output)
        EXPECT_EQ(word, 0xdeadbeef);
      EXPECT_EQ(fn(cpu | exact, 0U, 0, nullptr, &carry, nullptr, nullptr, UINT32_MAX), GOC_SUCCESS);
      EXPECT_EQ(carry, 0u);
      carry = 0x12345678;
      EXPECT_EQ(
          fn(cpu | 2 * GOC_SEMANTICS_EXACT_EMPIRICAL, 0, 0, nullptr, &carry, nullptr, nullptr, 0),
          GOC_SUCCESS);
      EXPECT_EQ(carry, 0u);
    }
}

TEST(Carry, DppMasksModifiersAndScalarVectorAliases) {
  std::mt19937 random(186397);
  uint32_t initial[3][34];
  for (auto &reg : initial)
    for (auto &word : reg)
      word = random();
  for (unsigned op = 0; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, exact})
        for (bool clamp : {false, true})
          for (auto descriptor : goc_test::dpp_modes)
            for (auto exec_mask : rdna4_exec_masks())
              for (unsigned source_b : {0u, 1u})
                for (unsigned target = 0; target < 3; ++target)
                  for (bool scalar_alias : {false, true}) {
                    uint32_t words[3][34], expected[3][34], carry = 0xdeadbeef, wanted_carry = 0;
                    uint32_t ci = exec_mask ^ 0xa5a5a5a5;
                    std::memcpy(words, initial, sizeof(words));
                    std::memcpy(expected, initial, sizeof(expected));
                    for (unsigned lane = 0; lane < 32; ++lane) {
                      int source = 0;
                      if (!goc_test::dpp_source(descriptor, exec_mask, lane, source))
                        continue;
                      uint32_t av = initial[0][lane + 1], bv = initial[source_b][lane + 1];
                      if (op % 3 == 2)
                        bv = source < 0 ? 0 : initial[source_b][source + 1];
                      else
                        av = source < 0 ? 0 : initial[0][source + 1];
                      auto gold = goc_test::carry_reference(op, av, bv, (ci >> lane) & 1, clamp);
                      expected[target][lane + 1] = gold.value;
                      wanted_carry |= uint32_t(gold.carry) << lane;
                    }
                    if (scalar_alias)
                      expected[target][14] = wanted_carry;
                    const uint32_t *a[] = {words[0] + 1}, *b[] = {words[source_b] + 1};
                    uint32_t *d[] = {words[target] + 1};
                    ASSERT_EQ(goc_test::carry_functions[op](
                                  cpu | semantics, exec_mask,
                                  descriptor | (clamp ? GOC_ALU_CLAMP : 0), d,
                                  scalar_alias ? words[target] + 14 : &carry, a, b, ci),
                              GOC_SUCCESS);
                    EXPECT_EQ(carry, scalar_alias ? 0xdeadbeef : wanted_carry);
                    ASSERT_EQ(std::memcmp(words, expected, sizeof(words)), 0)
                        << op << "/" << cpu << "/" << exec_mask << "/" << descriptor;
                  }
}

TEST(Carry, DppValidationAndZeroExec) {
  for (auto fn : goc_test::carry_functions)
    for (auto descriptor : goc_test::dpp_modes) {
      uint32_t carry = 0x12345678;
      for (auto invalid : {1ULL << 36, 1ULL})
        for (uint32_t exec_mask : {0U, UINT32_MAX}) {
          EXPECT_EQ(fn(0, exec_mask, descriptor | invalid, nullptr, &carry, nullptr, nullptr, 0),
                    GOC_ERROR_INVALID_FLAGS);
          EXPECT_EQ(carry, 0x12345678u);
        }
      EXPECT_EQ(fn(0, 0U, descriptor, nullptr, &carry, nullptr, nullptr, UINT32_MAX), GOC_SUCCESS);
      EXPECT_EQ(carry, 0u);
    }
}

// RX 9070: three carry-in masks, eight EXEC masks, six operations, both CLAMP
// modes and seven DPP descriptors. Each block hashes 32 VGPR words then carry.
TEST(Carry, DppHardwareCorpusAndHostFpState) {
  const uint32_t values[] = {0, 1, 2, 0x7ffffffe, 0x7fffffff, 0x80000000, 0xfffffffe, UINT32_MAX};
  const uint32_t masks[] = {0xffffffff, 0,          0xaaaaaaaa, 0x55555555,
                            1,          0x80000000, 0xffff,     0xffff0000};
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    std::fesetround(rounding);
    std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_INVALID | FE_INEXACT);
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, exact}) {
        uint64_t hash = goc_test::capture_hash_seed;
        for (uint32_t ci : {0u, UINT32_MAX, 0xa5a5a5a5u})
          for (auto exec_mask : masks)
            for (auto fn : goc_test::carry_functions)
              for (bool clamp : {false, true})
                for (auto descriptor : goc_test::dpp_modes) {
                  uint32_t av[32], bv[32], output[32], carry = 0x96969696;
                  for (unsigned lane = 0; lane < 32; ++lane) {
                    av[lane] = values[lane % 8];
                    bv[lane] = values[(lane + 3) % 8];
                    output[lane] = 0xdead0000u + lane;
                  }
                  const uint32_t *a[] = {av}, *b[] = {bv};
                  uint32_t *d[] = {output};
                  EXPECT_EQ(fn(cpu | semantics, exec_mask, descriptor | (clamp ? GOC_ALU_CLAMP : 0),
                               d, &carry, a, b, ci),
                            GOC_SUCCESS);
                  for (auto word : output)
                    hash = goc_test::capture_hash_word(hash, word);
                  hash = goc_test::capture_hash_word(hash, carry);
                }
        EXPECT_EQ(hash, 0x8c738ac66b29e6adULL) << cpu << "/" << semantics;
        EXPECT_EQ(std::fegetround(), rounding);
        EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_INVALID | FE_INEXACT);
      }
  }
}

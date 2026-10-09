// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_dpp_reference.h"
#include "rdna4_exec_masks.h"
#include "rdna4_fp8_conversion_reference.h"
#include "rdna4_fp8_narrow_hardware.h"
#include "rdna4_fp8_narrow_reference.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>
#include <vector>

namespace {

using Fn = decltype(&goc_rdna4_v_cvt_pk_fp8_f32);
const Fn functions[] = {goc_rdna4_v_cvt_pk_fp8_f32, goc_rdna4_v_cvt_pk_bf8_f32,
                        goc_rdna4_v_cvt_sr_fp8_f32, goc_rdna4_v_cvt_sr_bf8_f32};

uint32_t bits(float f) {
  uint32_t result;
  std::memcpy(&result, &f, sizeof(result));
  return result;
}

float value(uint32_t bits) {
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

} // namespace

TEST(Fp8Narrow, HardwareCapturedSpecialValuesAndStochasticAlignment) {
  for (const auto &capture : goc_test::fp8_narrow_captures)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned column = 0; column < 20; ++column) {
        unsigned op = ((column / 4) & 1) + (column % 4 >= 2 ? 2 : 0);
        uint32_t mode =
            column & 1 ? GOC_ALU_ABS_A | GOC_ALU_NEG_A | (op >= 2 ? GOC_CVT_BYTE_3 : GOC_ALU_ABS_B)
                       : 0;
        if (column >= 16) {
          op = (column - 16) & 1;
          mode = GOC_ALU_HIGH_D;
        }
        bool saturate = column < 16 ? column >= 8 : column >= 18;
        uint32_t av[32], bv[32], dv[32];
        std::fill_n(av, 32, capture.source);
        std::fill_n(bv, 32, op >= 2 ? capture.seed : capture.source ^ 0x80000000);
        std::fill_n(dv, 32, 0x12345678);
        const uint32_t *a[] = {av}, *b[] = {bv};
        uint32_t *d[] = {dv};
        ASSERT_EQ(functions[op](cpu | (saturate ? GOC_FP16_OVFL : 0), UINT32_MAX, mode, d, a, b),
                  GOC_SUCCESS);
        for (auto word : dv)
          ASSERT_EQ(word, capture.expected[column])
              << std::hex << capture.source << "/" << capture.seed << "/" << column << "/" << cpu;
      }
}

TEST(Fp8Narrow, RoundingBoundariesAndSeeds) {
  for (unsigned op = 0; op < 4; ++op) {
    std::vector<uint32_t> inputs = {0,          0x80000000, 1,          0x80000001,
                                    0x7f800000, 0xff800000, 0x7f800001, 0xffc12345};
    unsigned last = op & 1 ? 123 : 126;
    for (unsigned code = 0; code <= last; ++code) {
      float low = value(goc_test::fp8_conversion_reference(op & 1, uint8_t(code)));
      float high = code < last
                       ? value(goc_test::fp8_conversion_reference(op & 1, uint8_t(code + 1)))
                       : (op & 1 ? 65536.0f : 480.0f);
      for (float x : {low, (low + high) * 0.5f})
        for (int delta = -1; delta <= 1; ++delta) {
          uint32_t raw = bits(x) + uint32_t(delta);
          inputs.push_back(raw);
          inputs.push_back(raw ^ 0x80000000);
        }
    }
    for (int exponent = -65; exponent < 16; ++exponent)
      inputs.push_back(uint32_t(exponent + 127) << 23);
    std::mt19937 random(778 + op);
    for (unsigned i = 0; i < 1024; ++i)
      inputs.push_back(random());
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool sat : {false, true})
        for (uint32_t seed : {0u, 1u, 0x7fffffffu, 0x80000000u, 0xfffff000u, 0xffffffffu})
          for (unsigned start = 0; start < inputs.size(); start += 32) {
            uint32_t av[32], bv[32], dv[32];
            for (unsigned lane = 0; lane < 32; ++lane) {
              av[lane] = inputs[(start + lane) % inputs.size()];
              bv[lane] = op >= 2 ? seed : inputs[(start + lane + 1) % inputs.size()];
              dv[lane] = 0xa5c37691;
            }
            const uint32_t *a[] = {av}, *b[] = {bv};
            uint32_t *d[] = {dv};
            ASSERT_EQ(functions[op](cpu | (sat ? GOC_FP16_OVFL : 0), UINT32_MAX, 0, d, a, b),
                      GOC_SUCCESS);
            for (unsigned lane = 0; lane < 32; ++lane)
              ASSERT_EQ(dv[lane],
                        goc_test::fp8_narrow_result(op, av[lane], bv[lane], 0xa5c37691, 0, sat))
                  << op << "/" << cpu << "/" << sat << "/" << std::hex << av[lane] << "/"
                  << bv[lane];
          }
  }
}

TEST(Fp8Narrow, EveryModifierMaskAndWholeRegisterAlias) {
  std::mt19937 random(93321);
  for (unsigned op = 0; op < 4; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < (op >= 2 ? 16u : 32u); ++variant)
        for (bool sat : {false, true})
          for (uint64_t mask : rdna4_exec_masks())
            for (unsigned breg = 0; breg < 2; ++breg)
              for (unsigned dreg = 0; dreg < 3; ++dreg) {
                uint32_t storage[3][34], expected[3][34];
                for (unsigned reg = 0; reg < 3; ++reg)
                  for (unsigned word = 0; word < 34; ++word)
                    storage[reg][word] = expected[reg][word] = random();
                uint32_t mode = goc_test::fp8_narrow_mode(op, variant);
                for (unsigned lane = 0; lane < 32; ++lane)
                  if ((mask >> lane) & 1)
                    expected[dreg][lane + 1] = goc_test::fp8_narrow_result(
                        op, storage[0][lane + 1], storage[breg][lane + 1], storage[dreg][lane + 1],
                        mode, sat);
                const uint32_t *a[] = {storage[0] + 1}, *b[] = {storage[breg] + 1};
                uint32_t *d[] = {storage[dreg] + 1};
                ASSERT_EQ(functions[op](cpu | (sat ? GOC_FP16_OVFL : 0), mask, mode, d, a, b),
                          GOC_SUCCESS);
                for (unsigned reg = 0; reg < 3; ++reg)
                  for (unsigned word = 0; word < 34; ++word)
                    ASSERT_EQ(storage[reg][word], expected[reg][word])
                        << op << "/" << cpu << "/" << variant;
              }
}

TEST(Fp8Narrow, HostFloatingPointState) {
  fenv_t saved;
  ASSERT_EQ(std::fegetenv(&saved), 0);
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    EXPECT_EQ(std::fesetround(rounding), 0);
    for (unsigned op = 0; op < 4; ++op)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (unsigned variant = 0; variant < (op >= 2 ? 16u : 32u); ++variant) {
          uint32_t av[32], bv[32], dv[32], expected[32];
          uint32_t mode = goc_test::fp8_narrow_mode(op, variant);
          for (unsigned lane = 0; lane < 32; ++lane) {
            av[lane] = goc_test::fp8_narrow_captures[lane * 3].source;
            bv[lane] = 0x7fffffff;
            dv[lane] = 0xabcdef12;
            expected[lane] =
                goc_test::fp8_narrow_result(op, av[lane], bv[lane], dv[lane], mode, true);
          }
          const uint32_t *a[] = {av}, *b[] = {bv};
          uint32_t *d[] = {dv};
          EXPECT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
          EXPECT_EQ(std::feraiseexcept(FE_INEXACT | FE_INVALID), 0);
          int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
          EXPECT_EQ(functions[op](cpu | GOC_FP16_OVFL, UINT32_MAX, mode, d, a, b), GOC_SUCCESS);
          EXPECT_EQ(std::fegetround(), rounding);
          EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
          for (unsigned lane = 0; lane < 32; ++lane)
            EXPECT_EQ(dv[lane], expected[lane]);
        }
  }
  EXPECT_EQ(std::fesetenv(&saved), 0);
}

TEST(Fp8Narrow, ValidationAndSemanticFallback) {
  for (unsigned op = 0; op < 4; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t input[32] = {}, output[32];
      std::fill_n(output, 32, 0xdeadbeef);
      const uint32_t *a[] = {input};
      uint32_t *d[] = {output};
      uint32_t known = goc_test::fp8_narrow_mode(op, op >= 2 ? 15 : 31);
      for (unsigned bit = 0; bit < 32; ++bit)
        if (!(known & (1u << bit))) {
          EXPECT_EQ(functions[op](cpu, UINT32_MAX, 1u << bit, d, a, a), GOC_ERROR_INVALID_FLAGS);
          EXPECT_EQ(functions[op](cpu, 0, 1u << bit, nullptr, nullptr, nullptr),
                    GOC_ERROR_INVALID_FLAGS);
        }
      EXPECT_EQ(functions[op](cpu | (1ull << 63), UINT32_MAX, 0, d, a, a), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(functions[op](cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                              UINT32_MAX, 0, d, a, a),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
      for (auto word : output)
        EXPECT_EQ(word, 0xdeadbeef);
      EXPECT_EQ(functions[op](cpu, 0, known, nullptr, nullptr, nullptr), GOC_SUCCESS);
      EXPECT_EQ(functions[op](cpu, 0xffffffff00000000ull, known, nullptr, nullptr, nullptr),
                GOC_SUCCESS);
      EXPECT_EQ(functions[op](cpu | GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, 0, d, a, a),
                GOC_SUCCESS);
      for (auto word : output)
        EXPECT_EQ(word, op >= 2 ? 0xdeadbe00u : 0xdead0000u);
    }
}

TEST(Fp8Narrow, DppMasksAliasesAndGuards) {
  std::mt19937 random(93321);
  for (auto descriptor : goc_test::dpp_modes)
    for (unsigned op = 0; op < 4; ++op)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (unsigned variant : {0u, op >= 2 ? 8u : 16u, op >= 2 ? 15u : 31u})
          for (bool sat : {false, true})
            for (uint64_t mask : rdna4_exec_masks())
              for (unsigned breg = 0; breg < 2; ++breg)
                for (unsigned dreg = 0; dreg < 3; ++dreg) {
                  uint32_t storage[3][34], expected[3][34];
                  for (unsigned reg = 0; reg < 3; ++reg)
                    for (unsigned word = 0; word < 34; ++word)
                      storage[reg][word] = expected[reg][word] = random();
                  uint64_t mode = descriptor | goc_test::fp8_narrow_mode(op, variant);
                  for (unsigned lane = 0; lane < 32; ++lane) {
                    int source = 0;
                    if (goc_test::dpp_source(mode, uint32_t(mask), lane, source))
                      expected[dreg][lane + 1] = goc_test::fp8_narrow_result(
                          op, source < 0 ? 0 : storage[0][source + 1], storage[breg][lane + 1],
                          storage[dreg][lane + 1], uint32_t(mode), sat);
                  }
                  const uint32_t *a[] = {storage[0] + 1}, *b[] = {storage[breg] + 1};
                  uint32_t *d[] = {storage[dreg] + 1};
                  ASSERT_EQ(functions[op](cpu | (sat ? GOC_FP16_OVFL : 0), mask, mode, d, a, b),
                            GOC_SUCCESS);
                  for (unsigned reg = 0; reg < 3; ++reg)
                    for (unsigned word = 0; word < 34; ++word)
                      ASSERT_EQ(storage[reg][word], expected[reg][word])
                          << op << "/" << cpu << "/" << variant;
                }
}

// RX 9070: every modifier, both overflow modes, seven descriptors, eight masks.
TEST(Fp8Narrow, DppHardwareCorpus) {
  const uint32_t values[] = {0x3f880000, 0xbf900001, 0x43f00000, 0x47700000,
                             0x7f800001, 0xff800000, 0x33800000, 0x3f800001};
  const uint32_t masks[] = {0xffffffff, 0,          0xaaaaaaaa, 0x55555555,
                            1,          0x80000000, 0xffff,     0xffff0000};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (bool sat : {false, true})
      for (auto mask : masks)
        for (unsigned op = 0; op < 4; ++op)
          for (auto descriptor : goc_test::dpp_modes)
            for (unsigned variant = 0; variant < (op >= 2 ? 16u : 32u); ++variant) {
              uint32_t av[32], bv[32], output[32];
              for (unsigned lane = 0; lane < 32; ++lane) {
                av[lane] = values[lane % 8];
                bv[lane] = values[(lane + 3) % 8];
                output[lane] = 0xdead0000u + lane;
              }
              const uint32_t *a[] = {av}, *b[] = {bv};
              uint32_t *d[] = {output};
              ASSERT_EQ(functions[op](cpu | (sat ? GOC_FP16_OVFL : 0), mask,
                                      descriptor | goc_test::fp8_narrow_mode(op, variant), d, a, b),
                        GOC_SUCCESS);
              for (auto word : output)
                hash = (hash ^ word) * UINT64_C(1099511628211);
            }
    EXPECT_EQ(hash, UINT64_C(0x2e9896b81273f725)) << cpu;
  }
}

TEST(Fp8Narrow, DppValidation) {
  for (auto fn : functions)
    for (auto descriptor : goc_test::dpp_modes) {
      EXPECT_EQ(fn(0, 0, descriptor, nullptr, nullptr, nullptr), GOC_SUCCESS);
      for (auto invalid : {UINT64_C(1) << 36, uint64_t(GOC_ALU_CLAMP)})
        EXPECT_EQ(fn(0, 0, descriptor | invalid, nullptr, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
    }
}

TEST(Fp8Narrow, DppHostFloatingPointState) {
  fenv_t saved;
  ASSERT_EQ(std::fegetenv(&saved), 0);
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    EXPECT_EQ(std::fesetround(rounding), 0);
    for (auto descriptor : goc_test::dpp_modes)
      for (unsigned op = 0; op < 4; ++op)
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
          for (unsigned variant = 0; variant < (op >= 2 ? 16u : 32u); ++variant) {
            uint32_t av[32], bv[32], dv[32], expected[32];
            uint64_t mode = descriptor | goc_test::fp8_narrow_mode(op, variant);
            for (unsigned lane = 0; lane < 32; ++lane) {
              av[lane] = goc_test::fp8_narrow_captures[lane * 3].source;
              bv[lane] = 0x7fffffff;
              dv[lane] = 0xabcdef12;
            }
            for (unsigned lane = 0; lane < 32; ++lane) {
              int source = 0;
              expected[lane] = dv[lane];
              if (goc_test::dpp_source(mode, UINT32_MAX, lane, source))
                expected[lane] = goc_test::fp8_narrow_result(
                    op, source < 0 ? 0 : av[source], bv[lane], dv[lane], uint32_t(mode), true);
            }
            const uint32_t *a[] = {av}, *b[] = {bv};
            uint32_t *d[] = {dv};
            EXPECT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
            EXPECT_EQ(std::feraiseexcept(FE_INEXACT | FE_INVALID), 0);
            int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
            EXPECT_EQ(functions[op](cpu | GOC_FP16_OVFL, UINT32_MAX, mode, d, a, b), GOC_SUCCESS);
            EXPECT_EQ(std::fegetround(), rounding);
            EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
            for (unsigned lane = 0; lane < 32; ++lane)
              EXPECT_EQ(dv[lane], expected[lane]);
          }
  }
  EXPECT_EQ(std::fesetenv(&saved), 0);
}

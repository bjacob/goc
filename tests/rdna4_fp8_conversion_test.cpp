// SPDX-License-Identifier: MIT

#include "fp_environment.h"
#include "goc/goc.h"
#include "rdna4_dpp_reference.h"
#include "rdna4_exec_masks.h"
#include "rdna4_fp8_conversion_reference.h"

#include <algorithm>
#include <cfenv>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_rdna4_v_cvt_f32_fp8);
const Fn functions[] = {goc_rdna4_v_cvt_f32_fp8, goc_rdna4_v_cvt_f32_bf8,
                        goc_rdna4_v_cvt_pk_f32_fp8, goc_rdna4_v_cvt_pk_f32_bf8};

::testing::AssertionResult check(unsigned op, uint64_t cpu, unsigned selection,
                                 const uint32_t input[32]) {
  uint32_t output[2][32];
  const uint32_t *a[] = {input};
  uint32_t *d[] = {output[0], output[1]};
  bool packed = op >= 2;
  int status =
      functions[op](cpu, UINT32_MAX, goc_test::fp8_conversion_mode(packed, selection), d, a);
  if (status != GOC_SUCCESS)
    return ::testing::AssertionFailure() << "status " << status;
  unsigned shift = selection * (packed ? 16 : 8);
  for (unsigned reg = 0; reg < (packed ? 2u : 1u); ++reg)
    for (unsigned lane = 0; lane < 32; ++lane) {
      uint32_t expected =
          goc_test::fp8_conversion_reference(op & 1, uint8_t(input[lane] >> (shift + 8 * reg)));
      if (output[reg][lane] != expected)
        return ::testing::AssertionFailure()
               << op << "/" << cpu << "/" << selection << "/" << lane << std::hex << " input "
               << input[lane] << " expected " << expected << " actual " << output[reg][lane];
    }
  return ::testing::AssertionSuccess();
}

} // namespace

TEST(Fp8Conversion, LiteralFormatBoundaries) {
  const uint8_t codes[] = {0,    0x80, 1,    0x81, 7,    8,    0x38, 0x3c,
                           0x78, 0x7b, 0x7c, 0x7d, 0x7e, 0x7f, 0xfc, 0xff};
  const uint32_t expected[2][16] = {{0, 0x80000000, 0x3b000000, 0xbb000000, 0x3c600000, 0x3c800000,
                                     0x3f800000, 0x3fc00000, 0x43800000, 0x43b00000, 0x43c00000,
                                     0x43d00000, 0x43e00000, 0x7fc00000, 0xc3c00000, 0xffc00000},
                                    {0, 0x80000000, 0x37800000, 0xb7800000, 0x38e00000, 0x39000000,
                                     0x3f000000, 0x3f800000, 0x47000000, 0x47600000, 0x7f800000,
                                     0x7fc00000, 0x7fc00000, 0x7fc00000, 0xff800000, 0xffc00000}};
  for (unsigned op = 0; op < 4; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned select = 0; select < (op >= 2 ? 2u : 4u); ++select) {
        uint32_t input[32], output[2][32];
        for (unsigned lane = 0; lane < 32; ++lane)
          input[lane] = uint32_t(codes[lane % 16]) * 0x01010101u;
        const uint32_t *a[] = {input};
        uint32_t *d[] = {output[0], output[1]};
        ASSERT_EQ(
            functions[op](cpu, UINT32_MAX, goc_test::fp8_conversion_mode(op >= 2, select), d, a),
            GOC_SUCCESS);
        for (unsigned reg = 0; reg < (op >= 2 ? 2u : 1u); ++reg)
          for (unsigned lane = 0; lane < 32; ++lane)
            ASSERT_EQ(output[reg][lane], expected[op & 1][lane % 16]);
      }
}

TEST(Fp8Conversion, EveryCodeByteAndUnselectedBit) {
  for (unsigned op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned select = 0; select < 4; ++select)
        for (unsigned start = 0; start < 256; start += 32)
          for (unsigned noise = 0; noise < 33; ++noise) {
            uint32_t input[32];
            for (unsigned lane = 0; lane < 32; ++lane) {
              uint32_t unused = noise < 32 ? 1u << noise : UINT32_MAX;
              input[lane] = (unused & ~(255u << (8 * select))) | ((start + lane) << (8 * select));
            }
            ASSERT_TRUE(check(op, cpu, select, input));
          }
}

TEST(Fp8Conversion, EveryPackedPairAndHalfSelector) {
  for (unsigned op = 2; op < 4; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned select = 0; select < 2; ++select)
        for (unsigned start = 0; start < 65536; start += 32) {
          uint32_t input[32];
          for (unsigned lane = 0; lane < 32; ++lane)
            input[lane] = (start + lane) | ((65535 - start - lane) << 16);
          ASSERT_TRUE(check(op, cpu, select, input));
        }
}

TEST(Fp8Conversion, EverySelectorMaskAndDestinationAliasLayout) {
  std::mt19937 random(199);
  for (unsigned op = 0; op < 4; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned select = 0; select < (op >= 2 ? 2u : 4u); ++select)
        for (uint32_t mask : rdna4_exec_masks())
          for (unsigned low = 0; low < 3; ++low)
            for (unsigned high = 0; high < (op >= 2 ? 3u : 1u); ++high) {
              uint32_t storage[3][34], expected[3][34];
              for (unsigned reg = 0; reg < 3; ++reg)
                for (unsigned word = 0; word < 34; ++word)
                  storage[reg][word] = expected[reg][word] = random();
              unsigned shift = select * (op >= 2 ? 16 : 8);
              for (unsigned lane = 0; lane < 32; ++lane)
                if ((mask >> lane) & 1) {
                  uint32_t raw = storage[0][lane + 1] >> shift;
                  expected[low][lane + 1] =
                      goc_test::fp8_conversion_reference(op & 1, uint8_t(raw));
                  if (op >= 2)
                    expected[high][lane + 1] =
                        goc_test::fp8_conversion_reference(op & 1, uint8_t(raw >> 8));
                }
              const uint32_t *a[] = {storage[0] + 1};
              uint32_t *d[] = {storage[low] + 1, storage[high] + 1};
              ASSERT_EQ(
                  functions[op](cpu, mask, goc_test::fp8_conversion_mode(op >= 2, select), d, a),
                  GOC_SUCCESS);
              for (unsigned reg = 0; reg < 3; ++reg)
                for (unsigned word = 0; word < 34; ++word)
                  ASSERT_EQ(storage[reg][word], expected[reg][word])
                      << op << "/" << select << "/" << cpu;
            }
}

TEST(Fp8Conversion, EveryCodePreservesHostEnvironment) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
    EXPECT_EQ(std::fesetround(rounding), 0);
    EXPECT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
    EXPECT_EQ(std::feraiseexcept(FE_INVALID | FE_INEXACT), 0);
    int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
    for (unsigned op = 0; op < 4; ++op)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (unsigned start = 0; start < 256; start += 32) {
          uint32_t input[32];
          for (unsigned lane = 0; lane < 32; ++lane)
            input[lane] = (start + lane) * 0x01010101u;
          EXPECT_TRUE(check(op, cpu, op >= 2 ? 1 : 3, input));
          EXPECT_EQ(std::fegetround(), rounding);
          EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
        }
  }
}

TEST(Fp8Conversion, ValidationAndSemanticFallback) {
  for (unsigned op = 0; op < 4; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t input[32] = {}, output[2][32];
      std::fill_n(output[0], 32, 0xdeadbeef);
      std::fill_n(output[1], 32, 0xdeadbeef);
      const uint32_t *a[] = {input};
      uint32_t *d[] = {output[0], output[1]};
      uint32_t known = op >= 2 ? GOC_ALU_HIGH_A : GOC_CVT_BYTE_3;
      for (unsigned bit = 0; bit < 32; ++bit)
        if (!(known & (1u << bit))) {
          EXPECT_EQ(functions[op](cpu, UINT32_MAX, 1u << bit, d, a), GOC_ERROR_INVALID_FLAGS);
          EXPECT_EQ(functions[op](cpu, 0, 1u << bit, nullptr, nullptr), GOC_ERROR_INVALID_FLAGS);
        }
      EXPECT_EQ(functions[op](cpu | (UINT64_C(1) << 63), UINT32_MAX, 0, d, a),
                GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(functions[op](cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                              UINT32_MAX, 0, d, a),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
      for (auto &reg : output)
        for (auto word : reg)
          EXPECT_EQ(word, 0xdeadbeefu);
      EXPECT_EQ(functions[op](cpu, UINT32_C(0), 0, nullptr, nullptr), GOC_SUCCESS);
      for (unsigned sem = 0; sem < 4; ++sem)
        ASSERT_TRUE(check(op, cpu | (uint64_t(sem) << 16) | GOC_FP16_OVFL, 0, input));
    }
}

TEST(Fp8Conversion, DppSelectorsMasksAliasesAndGuards) {
  std::mt19937 random(8359);
  for (unsigned op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (auto descriptor : goc_test::dpp_modes)
        for (unsigned select = 0; select < 4; ++select)
          for (auto mask : rdna4_exec_masks())
            for (bool alias : {false, true}) {
              uint32_t storage[2][34], original[2][34];
              for (unsigned reg = 0; reg < 2; ++reg)
                for (unsigned word = 0; word < 34; ++word)
                  storage[reg][word] = original[reg][word] = random();
              const uint32_t *a[] = {storage[0] + 1};
              uint32_t *d[] = {storage[alias ? 0 : 1] + 1};
              uint64_t mode = descriptor | goc_test::fp8_conversion_mode(false, select);
              ASSERT_EQ(functions[op](cpu, mask, mode, d, a), GOC_SUCCESS);
              for (unsigned reg = 0; reg < 2; ++reg)
                for (unsigned word = 0; word < 34; ++word) {
                  uint32_t expected = original[reg][word];
                  int source = 0;
                  if (reg == unsigned(alias ? 0 : 1) && word > 0 && word <= 32 &&
                      goc_test::dpp_source(mode, mask, word - 1, source)) {
                    uint32_t raw = source < 0 ? 0 : original[0][source + 1];
                    expected = goc_test::fp8_conversion_reference(op, uint8_t(raw >> (select * 8)));
                  }
                  ASSERT_EQ(storage[reg][word], expected) << op << "/" << cpu << "/" << mode;
                }
            }
}

TEST(Fp8Conversion, DppValidation) {
  for (auto descriptor : goc_test::dpp_modes) {
    for (unsigned op = 0; op < 2; ++op) {
      EXPECT_EQ(functions[op](0, 0, descriptor, nullptr, nullptr), GOC_SUCCESS);
      for (auto invalid : {UINT64_C(1) << 36, uint64_t(GOC_ALU_NEG_A), uint64_t(GOC_ALU_ABS_A)})
        EXPECT_EQ(functions[op](0, 0, descriptor | invalid, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
    }
    // RDNA4's two-result widening forms have no DPP encoding.
    for (unsigned op = 2; op < 4; ++op)
      for (auto mask : {UINT64_C(0), UINT64_C(0xffffffff)})
        EXPECT_EQ(functions[op](0, mask, descriptor, nullptr, nullptr), GOC_ERROR_INVALID_FLAGS);
  }
}

// RX 9070 capture: every byte encoding and selector. Canonicalize FP32 NaNs.
TEST(Fp8Conversion, DppHardwareCorpusAndHostEnvironment) {
  const uint32_t masks[] = {0xffffffff, 0,          0xaaaaaaaa, 0x55555555,
                            1,          0x80000000, 0xffff,     0xffff0000};
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    EXPECT_EQ(std::fesetround(rounding), 0);
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      EXPECT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
      EXPECT_EQ(std::feraiseexcept(FE_INVALID | FE_INEXACT), 0);
      int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
      uint64_t hash = UINT64_C(14695981039346656037);
      for (unsigned batch = 0; batch < 8; ++batch)
        for (auto mask : masks)
          for (unsigned op = 0; op < 2; ++op)
            for (auto descriptor : goc_test::dpp_modes)
              for (unsigned select = 0; select < 4; ++select) {
                uint32_t input[32], output[32];
                for (unsigned lane = 0; lane < 32; ++lane) {
                  input[lane] = ((lane + batch * 32) * 0x01010101u) ^ 0x5aa55aa5u;
                  output[lane] = 0xdead0000u + lane;
                }
                const uint32_t *a[] = {input};
                uint32_t *d[] = {output};
                EXPECT_EQ(functions[op](cpu, mask,
                                        descriptor | goc_test::fp8_conversion_mode(false, select),
                                        d, a),
                          GOC_SUCCESS);
                for (auto word : output) {
                  if ((word & 0x7fffffff) > 0x7f800000)
                    word = 0x7fc00000;
                  hash = (hash ^ word) * UINT64_C(1099511628211);
                }
              }
      EXPECT_EQ(hash, UINT64_C(0xd937e5afc80bb725)) << cpu;
      EXPECT_EQ(std::fegetround(), rounding);
      EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
    }
  }
}

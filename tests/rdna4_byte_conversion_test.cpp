// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_byte_conversion_reference.h"
#include "rdna4_exec_masks.h"

#include <algorithm>
#include <cfenv>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_rdna4_v_cvt_f32_ubyte0);
const Fn functions[] = {goc_rdna4_v_cvt_f32_ubyte0, goc_rdna4_v_cvt_f32_ubyte1,
                        goc_rdna4_v_cvt_f32_ubyte2, goc_rdna4_v_cvt_f32_ubyte3};

} // namespace

TEST(ByteConversion, LiteralByteOrderAndOutputModifiers) {
  const uint32_t expected[4][8] = {{0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                                    0x00000000, 0x00000000, 0x00000000},
                                   {0x3f800000, 0x40000000, 0x40800000, 0x3f000000, 0x3f800000,
                                    0x3f800000, 0x3f800000, 0x3f000000},
                                   {0x43000000, 0x43800000, 0x44000000, 0x42800000, 0x3f800000,
                                    0x3f800000, 0x3f800000, 0x3f800000},
                                   {0x437f0000, 0x43ff0000, 0x447f0000, 0x42ff0000, 0x3f800000,
                                    0x3f800000, 0x3f800000, 0x3f800000}};
  for (unsigned byte = 0; byte < 4; ++byte)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < 8; ++variant) {
        uint32_t input[32], output[32];
        std::fill_n(input, 32, 0xff800100);
        const uint32_t *a[] = {input};
        uint32_t *d[] = {output};
        ASSERT_EQ(functions[byte](cpu, UINT32_MAX, goc_test::byte_conversion_mode(variant), d, a),
                  GOC_SUCCESS);
        for (auto word : output)
          ASSERT_EQ(word, expected[byte][variant]) << byte << "/" << variant << "/" << cpu;
      }
}

TEST(ByteConversion, EveryByteEveryModifierAndUnselectedBit) {
  std::mt19937 random(4791);
  for (unsigned byte = 0; byte < 4; ++byte)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < 8; ++variant)
        for (unsigned start = 0; start < 256; start += 32)
          for (unsigned noise = 0; noise < 34; ++noise) {
            uint32_t input[32], output[32];
            for (unsigned lane = 0; lane < 32; ++lane) {
              uint32_t unused = noise < 32    ? uint32_t(1) << noise
                                : noise == 32 ? UINT32_MAX
                                              : random();
              input[lane] = (unused & ~(255u << (8 * byte))) | ((start + lane) << (8 * byte));
            }
            uint32_t mode = goc_test::byte_conversion_mode(variant);
            const uint32_t *a[] = {input};
            uint32_t *d[] = {output};
            ASSERT_EQ(functions[byte](cpu, UINT32_MAX, mode, d, a), GOC_SUCCESS);
            for (unsigned lane = 0; lane < 32; ++lane)
              ASSERT_EQ(output[lane], goc_test::byte_conversion_reference(byte, input[lane], mode))
                  << byte << "/" << cpu << "/" << variant << "/" << lane;
          }
}

TEST(ByteConversion, EveryModifierMasksAliasesAndUnalignedStorage) {
  std::mt19937 random(2039);
  for (unsigned byte = 0; byte < 4; ++byte)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < 8; ++variant)
        for (uint64_t mask : rdna4_exec_masks())
          for (bool alias : {false, true}) {
            uint32_t storage[2][34], original[2][34];
            for (unsigned reg = 0; reg < 2; ++reg)
              for (unsigned word = 0; word < 34; ++word)
                storage[reg][word] = original[reg][word] = random();
            const uint32_t *a[] = {storage[0] + 1};
            uint32_t *d[] = {storage[alias ? 0 : 1] + 1};
            uint32_t mode = goc_test::byte_conversion_mode(variant);
            ASSERT_EQ(functions[byte](cpu, mask, mode, d, a), GOC_SUCCESS);
            for (unsigned reg = 0; reg < 2; ++reg)
              for (unsigned word = 0; word < 34; ++word) {
                uint32_t expected = original[reg][word];
                if (reg == unsigned(alias ? 0 : 1) && word > 0 && word <= 32 &&
                    ((mask >> (word - 1)) & 1))
                  expected = goc_test::byte_conversion_reference(byte, original[0][word], mode);
                ASSERT_EQ(storage[reg][word], expected) << byte << "/" << cpu << "/" << variant;
              }
          }
}

TEST(ByteConversion, HostRoundingAndExceptionsArePreserved) {
  fenv_t saved;
  ASSERT_EQ(std::fegetenv(&saved), 0);
  for (int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
    EXPECT_EQ(std::fesetround(rounding), 0);
    EXPECT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
    EXPECT_EQ(std::feraiseexcept(FE_INVALID | FE_INEXACT), 0);
    int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
    for (auto cpu = UINT64_C(0); cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned byte = 0; byte < 4; ++byte)
        for (unsigned variant = 0; variant < 8; ++variant)
          for (unsigned start = 0; start < 256; start += 32) {
            uint32_t input[32], output[32];
            for (unsigned lane = 0; lane < 32; ++lane)
              input[lane] = (start + lane) * 0x01010101u;
            const uint32_t *a[] = {input};
            uint32_t *d[] = {output};
            uint32_t mode = goc_test::byte_conversion_mode(variant);
            EXPECT_EQ(functions[byte](cpu, UINT32_MAX, mode, d, a), GOC_SUCCESS);
            EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
            EXPECT_EQ(std::fegetround(), rounding);
            for (unsigned lane = 0; lane < 32; ++lane)
              EXPECT_EQ(output[lane], goc_test::byte_conversion_reference(byte, input[lane], mode));
          }
  }
  EXPECT_EQ(std::fesetenv(&saved), 0);
}

TEST(ByteConversion, ValidationAndSemanticFallback) {
  for (auto fn : functions)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t input[32] = {}, output[32];
      std::fill_n(output, 32, 0xdeadbeef);
      const uint32_t *a[] = {input};
      uint32_t *d[] = {output};
      for (unsigned bit = 0; bit < 32; ++bit)
        if (!(uint32_t(1) << bit & (GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP))) {
          EXPECT_EQ(fn(cpu, UINT32_MAX, uint32_t(1) << bit, d, a), GOC_ERROR_INVALID_FLAGS);
          EXPECT_EQ(fn(cpu, 0, uint32_t(1) << bit, nullptr, nullptr), GOC_ERROR_INVALID_FLAGS);
        }
      EXPECT_EQ(fn(cpu | (UINT64_C(1) << 63), UINT32_MAX, 0, d, a), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, UINT32_MAX, 0, d, a),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
      for (auto word : output)
        EXPECT_EQ(word, 0xdeadbeef);
      EXPECT_EQ(fn(cpu, UINT64_C(0xffffffff00000000), 0, nullptr, nullptr), GOC_SUCCESS);
      for (unsigned semantics = 0; semantics < 4; ++semantics) {
        EXPECT_EQ(fn(cpu | (uint64_t(semantics) << 16) | GOC_FP16_OVFL, UINT32_MAX, 0, d, a),
                  GOC_SUCCESS);
        for (auto word : output)
          EXPECT_EQ(word, 0u);
      }
    }
}

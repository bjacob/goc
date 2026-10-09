// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "rdna4_conversion32_reference.h"
#include "rdna4_dpp_reference.h"
#include "rdna4_exec_masks.h"

#include <algorithm>
#include <cfenv>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_rdna4_v_cvt_f32_i32);
const Fn functions[] = {goc_rdna4_v_cvt_f32_i32,         goc_rdna4_v_cvt_f32_u32,
                        goc_rdna4_v_cvt_i32_f32,         goc_rdna4_v_cvt_u32_f32,
                        goc_rdna4_v_cvt_nearest_i32_f32, goc_rdna4_v_cvt_floor_i32_f32};

::testing::AssertionResult check(int op, uint64_t flags, uint32_t mode, const uint32_t *input) {
  uint32_t output[32];
  const uint32_t *a[] = {input};
  uint32_t *d[] = {output};
  int status = functions[op](flags, UINT32_MAX, mode, d, a);
  if (status != GOC_SUCCESS)
    return ::testing::AssertionFailure() << "status " << status;
  for (int lane = 0; lane < 32; ++lane) {
    uint32_t expected = goc_test::conversion32_reference(op, input[lane], mode);
    if (output[lane] != expected)
      return ::testing::AssertionFailure()
             << op << "/" << flags << "/" << mode << "/" << lane << std::hex << " input "
             << input[lane] << " expected " << expected << " got " << output[lane];
  }
  return ::testing::AssertionSuccess();
}

} // namespace

TEST(Conversion32, LiteralRoundingAndSaturationWitnesses) {
  const uint32_t inputs[] = {0x3f000000, 0xbf000000, 0x3fc00000, 0xbfc00000, 0x40200000, 0xc0200000,
                             1,          0x80000001, 0x7f800000, 0xff800000, 0x7fc12345, 0xff800001,
                             0x4effffff, 0x4f000000, 0xcf000000, 0x4f7fffff, 0x4f800000};
  const uint32_t expected[4][17] = {
      {0, 0, 1, 0xffffffff, 2, 0xfffffffe, 0, 0, 0x7fffffff, 0x80000000, 0, 0, 0x7fffff80,
       0x7fffffff, 0x80000000, 0x7fffffff, 0x7fffffff},
      {0, 0, 1, 0, 2, 0, 0, 0, 0xffffffff, 0, 0, 0, 0x7fffff80, 0x80000000, 0, 0xffffff00,
       0xffffffff},
      {1, 0, 2, 0xffffffff, 3, 0xfffffffe, 0, 0, 0x7fffffff, 0x80000000, 0x7fffffff, 0x80000000,
       0x7fffff80, 0x7fffffff, 0x80000000, 0x7fffffff, 0x7fffffff},
      {0, 0xffffffff, 1, 0xfffffffe, 2, 0xfffffffd, 0, 0xffffffff, 0x7fffffff, 0x80000000,
       0x7fffffff, 0x80000000, 0x7fffff80, 0x7fffffff, 0x80000000, 0x7fffffff, 0x7fffffff}};
  for (int op = 2; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t input[32], output[32];
      for (int lane = 0; lane < 32; ++lane)
        input[lane] = inputs[lane % 17];
      const uint32_t *a[] = {input};
      uint32_t *d[] = {output};
      ASSERT_EQ(functions[op](cpu, UINT32_MAX, 0, d, a), GOC_SUCCESS);
      for (int lane = 0; lane < 32; ++lane)
        EXPECT_EQ(output[lane], expected[op - 2][lane % 17]) << op << "/" << cpu << "/" << lane;
    }
  const uint32_t integer_inputs[] = {16777217,   16777219,   0xffffffff,
                                     0x80000000, 0x7fffffff, 0x80000081};
  const uint32_t encoded[2][6] = {
      {0x4b800000, 0x4b800002, 0xbf800000, 0xcf000000, 0x4f000000, 0xceffffff},
      {0x4b800000, 0x4b800002, 0x4f800000, 0x4f000000, 0x4f000000, 0x4f000001}};
  for (int op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t input[32], output[32];
      for (int lane = 0; lane < 32; ++lane)
        input[lane] = integer_inputs[lane % 6];
      const uint32_t *a[] = {input};
      uint32_t *d[] = {output};
      ASSERT_EQ(functions[op](cpu, UINT32_MAX, 0, d, a), GOC_SUCCESS);
      for (int lane = 0; lane < 32; ++lane)
        EXPECT_EQ(output[lane], encoded[op][lane % 6]) << op << "/" << cpu << "/" << lane;
    }
}

TEST(Conversion32, PrecisionBoundariesAndEveryModifier) {
  for (int op = 0; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < goc_test::conversion32_modes(op); ++variant) {
        uint32_t mode = goc_test::conversion32_mode(op, variant);
        for (unsigned exponent = 0; exponent < (op < 2 ? 32u : 256u); ++exponent)
          for (unsigned sign = 0; sign < 2; ++sign) {
            uint32_t input[32];
            for (unsigned lane = 0; lane < 32; ++lane) {
              uint32_t base = op < 2 ? uint32_t(1) << exponent : exponent << 23;
              input[lane] = (base + lane - 16) ^ (sign << 31);
            }
            ASSERT_TRUE(check(op, cpu, mode, input));
          }
      }
}

TEST(Conversion32, RandomFullWordsAndEveryModifier) {
  for (int op = 0; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < goc_test::conversion32_modes(op); ++variant) {
        std::mt19937 random(23187);
        for (int batch = 0; batch < 256; ++batch) {
          uint32_t input[32];
          for (auto &word : input)
            word = random();
          ASSERT_TRUE(check(op, cpu, goc_test::conversion32_mode(op, variant), input));
        }
      }
}

TEST(Conversion32, MasksAliasesAndUnalignedStorage) {
  std::mt19937 random(162);
  for (int op = 0; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < goc_test::conversion32_modes(op); ++variant)
        for (uint32_t mask : rdna4_exec_masks())
          for (bool alias : {false, true}) {
            uint32_t storage[2][35], original[2][35];
            for (auto &reg : storage)
              for (auto &word : reg)
                word = random();
            std::copy(&storage[0][0], &storage[0][0] + 35, original[0]);
            std::copy(&storage[1][0], &storage[1][0] + 35, original[1]);
            const uint32_t *a[] = {storage[0] + 1};
            uint32_t *d[] = {storage[alias ? 0 : 1] + 1};
            uint32_t mode = goc_test::conversion32_mode(op, variant);
            ASSERT_EQ(functions[op](cpu, mask, mode, d, a), GOC_SUCCESS);
            for (int reg = 0; reg < 2; ++reg)
              for (int word = 0; word < 35; ++word) {
                uint32_t expected = original[reg][word];
                if (reg == (alias ? 0 : 1) && word >= 1 && word <= 32 && ((mask >> (word - 1)) & 1))
                  expected = goc_test::conversion32_reference(op, original[0][word], mode);
                ASSERT_EQ(storage[reg][word], expected);
              }
          }
}

TEST(Conversion32, ValidationAndSemanticFallback) {
  for (int op = 0; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t input[32] = {}, output[32];
      const uint32_t *a[] = {input};
      uint32_t *d[] = {output};
      std::fill_n(output, 32, 0xdeadbeef);
      uint32_t known =
          op < 2 ? GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP
                 : GOC_ALU_NEG_A | GOC_ALU_ABS_A | GOC_ALU_CLAMP | (op < 4 ? GOC_ALU_OMOD_HALF : 0);
      for (unsigned bit = 0; bit < 32; ++bit)
        if (!(known & (uint32_t(1) << bit))) {
          EXPECT_EQ(functions[op](cpu, UINT32_MAX, uint32_t(1) << bit, d, a),
                    GOC_ERROR_INVALID_FLAGS);
          EXPECT_EQ(functions[op](cpu, 0, uint32_t(1) << bit, nullptr, nullptr),
                    GOC_ERROR_INVALID_FLAGS);
        }
      EXPECT_EQ(functions[op](cpu | (UINT64_C(1) << 63), UINT32_MAX, 0, d, a),
                GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(functions[op](cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                              UINT32_MAX, 0, d, a),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
      for (auto word : output)
        EXPECT_EQ(word, 0xdeadbeef);
      EXPECT_EQ(functions[op](cpu, UINT32_C(0), 0, nullptr, nullptr), GOC_SUCCESS);
      for (unsigned sem = 1; sem < 4; ++sem)
        ASSERT_TRUE(check(op, cpu | (uint64_t(sem) << 16), 0, input));
    }
}

TEST(Conversion32, FloatToIntegerIgnoresHostRounding) {
  goc_test::ScopedFpEnvironment original;
  ASSERT_TRUE(original.saved());
  uint32_t input[32];
  const uint32_t values[] = {0x3f000000, 0xbf000000, 0x3fc00000, 0xbfc00000,
                             1,          0x80000001, 0x4f000000, 0x4f800000};
  for (int lane = 0; lane < 32; ++lane)
    input[lane] = values[lane % 8];
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    EXPECT_EQ(std::fesetround(rounding), 0);
    for (int op = 2; op < 6; ++op)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
        EXPECT_TRUE(check(op, cpu, 0, input));
        EXPECT_EQ(std::fegetround(), rounding);
      }
  }
}

TEST(Conversion32, DppModifiersMasksAliasesAndRandomWords) {
  std::mt19937 random(162);
  for (int op = 0; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint64_t descriptor : goc_test::dpp_modes)
        for (unsigned variant = 0; variant < goc_test::conversion32_modes(op); ++variant)
          for (uint32_t mask : rdna4_exec_masks())
            for (bool alias : {false, true}) {
              uint32_t storage[2][35], original[2][35];
              for (auto &reg : storage)
                for (auto &word : reg)
                  word = random();
              std::copy(&storage[0][0], &storage[0][0] + 35, original[0]);
              std::copy(&storage[1][0], &storage[1][0] + 35, original[1]);
              const uint32_t *a[] = {storage[0] + 1};
              uint32_t *d[] = {storage[alias ? 0 : 1] + 1};
              uint32_t mode = goc_test::conversion32_mode(op, variant);
              ASSERT_EQ(functions[op](cpu, mask, descriptor | mode, d, a), GOC_SUCCESS);
              for (int reg = 0; reg < 2; ++reg)
                for (int word = 0; word < 35; ++word) {
                  uint32_t expected = original[reg][word];
                  int source;
                  if (reg == (alias ? 0 : 1) && word >= 1 && word <= 32 &&
                      goc_test::dpp_source(descriptor, mask, word - 1, source))
                    expected = goc_test::conversion32_reference(
                        op, source < 0 ? 0 : original[0][source + 1], mode);
                  ASSERT_EQ(storage[reg][word], expected);
                }
            }
}

TEST(Conversion32, HardwareOrdinaryAndDppCorpus) {
  const uint32_t values[] = {0,          1,          0xffffffff, 0x80000000, 0x01000001, 0x7fffffff,
                             0x3f000000, 0xbf000000, 0x3fc00000, 0xbfc00000, 0x4effffff, 0xcf000000,
                             0x4f800000, 0xff800000, 0x7f800001, 0x80000001};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (bool dpp : {false, true}) {
      uint64_t hash = goc_test::capture_hash_seed;
      for (uint32_t mask :
           {0xffffffffu, 0u, 0xaaaaaaaau, 0x55555555u, 1u, 0x80000000u, 0xffffu, 0xffff0000u})
        for (unsigned op = 0; op < 6; ++op)
          for (unsigned index = 0; index < (dpp ? 7u : 1u); ++index)
            for (unsigned variant = 0; variant < 8; ++variant) {
              uint64_t descriptor = dpp ? goc_test::dpp_modes[index] : 0;
              uint32_t a[32], d[32];
              uint32_t mode = goc_test::conversion32_mode(op, variant);
              for (unsigned lane = 0; lane < 32; ++lane) {
                a[lane] = values[lane % 16];
                d[lane] = 0xdead0000u + lane;
              }
              auto pa = a, pd = d;
              ASSERT_EQ(functions[op](cpu, mask, descriptor | mode, &pd, &pa), GOC_SUCCESS);
              for (unsigned lane = 0; lane < 32; ++lane) {
                int source = lane;
                uint32_t want = 0xdead0000u + lane;
                if (dpp ? goc_test::dpp_source(descriptor, mask, lane, source)
                        : ((mask >> lane) & 1))
                  want = goc_test::conversion32_reference(op, source < 0 ? 0u : a[source], mode);
                EXPECT_EQ(d[lane], want);
                hash = goc_test::capture_hash_word(hash, d[lane]);
              }
            }
      EXPECT_EQ(hash, dpp ? UINT64_C(0x4fc3a683d5b1e3b5) : UINT64_C(0x87a638b369ca680d));
    }
}

TEST(Conversion32, DppValidation) {
  for (unsigned op = 0; op < 6; ++op)
    for (uint64_t descriptor : goc_test::dpp_modes) {
      EXPECT_EQ(functions[op](0, 0, descriptor, nullptr, nullptr), GOC_SUCCESS);
      EXPECT_EQ(functions[op](0, UINT32_MAX, descriptor | GOC_ALU_HIGH_A, nullptr, nullptr),
                GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(functions[op](0, UINT32_MAX, descriptor | (UINT64_C(1) << 36), nullptr, nullptr),
                GOC_ERROR_INVALID_FLAGS);
      if (op < 2) {
        EXPECT_EQ(functions[op](0, UINT32_MAX, descriptor | GOC_ALU_NEG_A, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
      }
      if (op >= 4) {
        EXPECT_EQ(functions[op](0, UINT32_MAX, descriptor | GOC_ALU_OMOD_2, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
      }
      EXPECT_EQ(functions[op](GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, UINT32_MAX,
                              descriptor, nullptr, nullptr),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
    }
}

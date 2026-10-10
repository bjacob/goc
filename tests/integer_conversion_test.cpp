// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "dpp_reference.h"
#include "exec_masks.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "integer_conversion_reference.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <random>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_v_cvt_pk_i16_i32);
const Fn functions[] = {
    [](uint64_t f, uint32_t exec_mask, uint64_t i, uint32_t *const *d, const uint32_t *const *a,
       const uint32_t *const *) { return goc_v_cvt_i32_i16(f, exec_mask, i, d, a); },
    [](uint64_t f, uint32_t exec_mask, uint64_t i, uint32_t *const *d, const uint32_t *const *a,
       const uint32_t *const *) { return goc_v_cvt_u32_u16(f, exec_mask, i, d, a); },
    goc_v_cvt_pk_i16_i32, goc_v_cvt_pk_u16_u32};

} // namespace

TEST(IntegerConversion, LiteralSaturationAndHalfOrder) {
  const uint32_t inputs[] = {0,       1,          0xffffffff, 0x7fff,     0x8000,    0xffff,
                             0x10000, 0xffff7fff, 0xffff8000, 0x80000000, 0x7fffffff};
  const uint32_t signed_results[] = {0,      1,      0xffff, 0x7fff, 0x7fff, 0x7fff,
                                     0x7fff, 0x8000, 0x8000, 0x8000, 0x7fff};
  const uint32_t unsigned_results[] = {0,      1,      0xffff, 0x7fff, 0x8000, 0xffff,
                                       0xffff, 0xffff, 0xffff, 0xffff, 0xffff};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned i = 0; i < 11; ++i)
      for (unsigned j = 0; j < 11; ++j) {
        uint32_t av[32], bv[32], output[32];
        std::fill_n(av, 32, inputs[i]);
        std::fill_n(bv, 32, inputs[j]);
        const uint32_t *a[] = {av}, *b[] = {bv};
        uint32_t *d[] = {output};
        for (unsigned op = 2; op < 4; ++op) {
          ASSERT_EQ(functions[op](cpu, UINT32_MAX, 0, d, a, b), GOC_SUCCESS);
          const uint32_t *values = op == 2 ? signed_results : unsigned_results;
          for (auto word : output)
            ASSERT_EQ(word, values[i] | (values[j] << 16));
        }
      }
}

TEST(IntegerConversion, EveryHalfAndSaturationBoundary) {
  for (unsigned op = 0; op < 4; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned select = 0; select < (op < 2 ? 2u : 1u); ++select)
        for (unsigned start = 0; start < 65536; start += 32) {
          uint32_t av[32], bv[32], output[32];
          for (unsigned lane = 0; lane < 32; ++lane) {
            av[lane] = op < 2 ? (start + lane) | ((65535 - start - lane) << 16) : start + lane;
            bv[lane] = 0u - (start + lane);
          }
          const uint32_t *a[] = {av}, *b[] = {bv};
          uint32_t *d[] = {output};
          uint32_t mode = select ? GOC_ALU_HIGH_A : 0;
          ASSERT_EQ(functions[op](cpu, UINT32_MAX, mode, d, a, b), GOC_SUCCESS);
          for (unsigned lane = 0; lane < 32; ++lane)
            ASSERT_EQ(output[lane],
                      goc_test::integer_conversion_reference(op, av[lane], bv[lane], mode));
        }
}

TEST(IntegerConversion, MasksAliasesAndUnalignedFullWords) {
  std::mt19937 random(8877);
  for (unsigned op = 0; op < 4; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned select = 0; select < (op < 2 ? 2u : 1u); ++select)
        for (uint32_t exec_mask : exec_masks())
          for (unsigned breg = 0; breg < (op >= 2 ? 2u : 1u); ++breg)
            for (unsigned dreg = 0; dreg < 3; ++dreg) {
              uint32_t storage[3][34], expected[3][34];
              for (unsigned reg = 0; reg < 3; ++reg)
                for (unsigned word = 0; word < 34; ++word)
                  storage[reg][word] = expected[reg][word] = random();
              uint32_t mode = select ? GOC_ALU_HIGH_A : 0;
              for (unsigned lane = 0; lane < 32; ++lane)
                if ((exec_mask >> lane) & 1)
                  expected[dreg][lane + 1] = goc_test::integer_conversion_reference(
                      op, storage[0][lane + 1], storage[breg][lane + 1], mode);
              const uint32_t *a[] = {storage[0] + 1}, *b[] = {storage[breg] + 1};
              uint32_t *d[] = {storage[dreg] + 1};
              ASSERT_EQ(functions[op](cpu | GOC_FP16_OVFL, exec_mask, mode, d, a, b), GOC_SUCCESS);
              for (unsigned reg = 0; reg < 3; ++reg)
                for (unsigned word = 0; word < 34; ++word)
                  ASSERT_EQ(storage[reg][word], expected[reg][word]);
            }
}

TEST(IntegerConversion, HostEnvironmentIsPreserved) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  std::mt19937 random(913);
  for (int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
    EXPECT_EQ(std::fesetround(rounding), 0);
    EXPECT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
    EXPECT_EQ(std::feraiseexcept(FE_INVALID | FE_INEXACT), 0);
    int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
    for (unsigned op = 0; op < 4; ++op)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
        uint32_t av[32], bv[32], output[32];
        for (unsigned lane = 0; lane < 32; ++lane) {
          av[lane] = random();
          bv[lane] = random();
        }
        const uint32_t *a[] = {av}, *b[] = {bv};
        uint32_t *d[] = {output};
        EXPECT_EQ(functions[op](cpu, UINT32_MAX, op < 2 ? GOC_ALU_HIGH_A : 0, d, a, b),
                  GOC_SUCCESS);
        EXPECT_EQ(std::fegetround(), rounding);
        EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
      }
  }
}

TEST(IntegerConversion, ValidationAndSemanticFallback) {
  for (unsigned op = 0; op < 4; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t input[32] = {}, output[32];
      std::fill_n(output, 32, 0xdeadbeef);
      const uint32_t *a[] = {input};
      uint32_t *d[] = {output};
      uint32_t known = op < 2 ? GOC_ALU_HIGH_A : 0;
      for (unsigned bit = 0; bit < 32; ++bit)
        if (!(known & (1u << bit))) {
          EXPECT_EQ(functions[op](cpu, UINT32_MAX, 1u << bit, d, a, a), GOC_ERROR_INVALID_FLAGS);
          EXPECT_EQ(functions[op](cpu, 0, 1u << bit, nullptr, nullptr, nullptr),
                    GOC_ERROR_INVALID_FLAGS);
        }
      EXPECT_EQ(functions[op](cpu | (1ULL << 63), UINT32_MAX, 0, d, a, a), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(functions[op](cpu | (2ULL << 16) | GOC_SEMANTICS_STRICT, UINT32_MAX, 0, d, a, a),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
      for (auto word : output)
        EXPECT_EQ(word, 0xdeadbeefu);
      EXPECT_EQ(functions[op](cpu, 0U, 0, nullptr, nullptr, nullptr), GOC_SUCCESS);
      for (unsigned sem = 0; sem < 4; ++sem) {
        EXPECT_EQ(functions[op](cpu | (uint64_t(sem) << 16), UINT32_MAX, 0, d, a, a), GOC_SUCCESS);
        for (auto word : output)
          EXPECT_EQ(word, 0u);
      }
    }
}

TEST(IntegerConversion, DppMasksAliasesAndUnalignedFullWords) {
  std::mt19937 random(8877);
  for (auto descriptor : goc_test::dpp_modes)
    for (unsigned op = 0; op < 4; ++op)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (unsigned select = 0; select < (op < 2 ? 2u : 1u); ++select)
          for (uint32_t exec_mask : exec_masks())
            for (unsigned breg = 0; breg < (op >= 2 ? 2u : 1u); ++breg)
              for (unsigned dreg = 0; dreg < 3; ++dreg) {
                uint32_t storage[3][34], expected[3][34];
                for (unsigned reg = 0; reg < 3; ++reg)
                  for (unsigned word = 0; word < 34; ++word)
                    storage[reg][word] = expected[reg][word] = random();
                uint64_t mode = descriptor | (select ? GOC_ALU_HIGH_A : 0);
                for (unsigned lane = 0; lane < 32; ++lane) {
                  int source = 0;
                  if (goc_test::dpp_source(mode, exec_mask, lane, source))
                    expected[dreg][lane + 1] = goc_test::integer_conversion_reference(
                        op, source < 0 ? 0 : storage[0][source + 1], storage[breg][lane + 1],
                        uint32_t(mode));
                }
                const uint32_t *a[] = {storage[0] + 1}, *b[] = {storage[breg] + 1};
                uint32_t *d[] = {storage[dreg] + 1};
                ASSERT_EQ(functions[op](cpu | GOC_FP16_OVFL, exec_mask, mode, d, a, b),
                          GOC_SUCCESS);
                for (unsigned reg = 0; reg < 3; ++reg)
                  for (unsigned word = 0; word < 34; ++word)
                    ASSERT_EQ(storage[reg][word], expected[reg][word]);
              }
}

// RX 9070 capture, including source half selectors and preserved destinations.
TEST(IntegerConversion, DppHardwareCorpus) {
  const uint32_t values[] = {0, 1, 0xffffffff, 0x7fff, 0x8000, 0x10000, 0xffff7fff, 0x80000000};
  const uint32_t masks[] = {0xffffffff, 0,          0xaaaaaaaa, 0x55555555,
                            1,          0x80000000, 0xffff,     0xffff0000};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint64_t hash = goc_test::capture_hash_seed;
    for (auto exec_mask : masks)
      for (unsigned op = 0; op < 4; ++op)
        for (auto descriptor : goc_test::dpp_modes)
          for (unsigned select = 0; select < (op < 2 ? 2u : 1u); ++select) {
            uint32_t av[32], bv[32], output[32];
            for (unsigned lane = 0; lane < 32; ++lane) {
              av[lane] = values[lane % 8];
              bv[lane] = values[(lane + 3) % 8];
              output[lane] = 0xdead0000u + lane;
            }
            const uint32_t *a[] = {av}, *b[] = {bv};
            uint32_t *d[] = {output};
            ASSERT_EQ(
                functions[op](cpu, exec_mask, descriptor | (select ? GOC_ALU_HIGH_A : 0), d, a, b),
                GOC_SUCCESS);
            for (auto word : output)
              hash = goc_test::capture_hash_word(hash, word);
          }
    EXPECT_EQ(hash, 0x1520b5b44ef7c66aULL) << cpu;
  }
}

TEST(IntegerConversion, DppValidation) {
  for (auto fn : functions)
    for (auto descriptor : goc_test::dpp_modes) {
      EXPECT_EQ(fn(0, 0, descriptor, nullptr, nullptr, nullptr), GOC_SUCCESS);
      for (auto invalid : std::initializer_list<uint64_t>{1ULL << 36, uint64_t(GOC_ALU_ABS_A),
                                                          uint64_t(GOC_ALU_CLAMP)})
        EXPECT_EQ(fn(0, 0, descriptor | invalid, nullptr, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
    }
}

TEST(IntegerConversion, Wave64ExactHalvesAndBoundaries) {
  const decltype(&goc_v_cvt_pk_i16_i32_wave64) wave64[] = {
      [](uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
         const uint32_t *const *a, const uint32_t *const *) {
        return goc_v_cvt_i32_i16_wave64(flags, exec_mask, mode, d, a);
      },
      [](uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
         const uint32_t *const *a, const uint32_t *const *) {
        return goc_v_cvt_u32_u16_wave64(flags, exec_mask, mode, d, a);
      },
      goc_v_cvt_pk_i16_i32_wave64, goc_v_cvt_pk_u16_u32_wave64};
  for (unsigned op = 0; op < 4; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned select = 0; select < (op < 2 ? 2U : 1U); ++select) {
        uint64_t flags = cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;
        uint64_t mode = select ? GOC_ALU_HIGH_A : 0;
        EXPECT_EQ(wave64[op](flags, 0, mode, nullptr, nullptr, nullptr), GOC_SUCCESS);
        EXPECT_EQ(wave64[op](flags, UINT64_MAX, 1ULL << 63, nullptr, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
        for (unsigned start = 0; start < 65536; start += 64) {
          uint32_t av[64], bv[64], output[64];
          for (unsigned lane = 0; lane < 64; ++lane) {
            unsigned i = start + lane;
            av[lane] = op < 2 ? i | ((65535 - i) << 16) : i;
            bv[lane] = 0U - i;
          }
          const uint32_t *a = av, *b = bv;
          uint32_t *d = output;
          ASSERT_EQ(wave64[op](flags, UINT64_MAX, mode, &d, &a, &b), GOC_SUCCESS);
          for (unsigned lane = 0; lane < 64; ++lane)
            ASSERT_EQ(output[lane],
                      goc_test::integer_conversion_reference(op, av[lane], bv[lane], mode));
        }
        for (auto route : goc_test::dpp_modes)
          for (unsigned rows = 0; rows < 16; ++rows)
            for (unsigned target = 0; target < 3; ++target) {
              uint32_t initial[3][66], words[3][66];
              for (unsigned lane = 0; lane < 66; ++lane) {
                initial[0][lane] = 0x7395a831U * lane;
                initial[1][lane] = 0x87654321U * lane;
                initial[2][lane] = 0xdeadbeef;
              }
              std::memcpy(words, initial, sizeof words);
              const uint32_t *a = words[0] + 1, *b = words[1] + 1;
              uint32_t *d = words[target] + 1;
              uint64_t descriptor = route;
              if (route & GOC_DPP16)
                descriptor = (route & ~GOC_DPP_ROW_MASK) | (uint64_t(rows) << GOC_DPP_ROW_SHIFT);
              uint64_t exec_mask = 0x87654321fedcba98ULL;
              ASSERT_EQ(wave64[op](flags, exec_mask, mode | descriptor, &d, &a, &b), GOC_SUCCESS);
              for (unsigned lane = 0; lane < 64; ++lane) {
                int source = 0;
                bool enabled = goc_test::dpp_source(descriptor, exec_mask, lane, source);
                uint32_t expected = initial[target][lane + 1];
                if (enabled)
                  expected = goc_test::integer_conversion_reference(
                      op, source < 0 ? 0 : initial[0][source + 1], initial[1][lane + 1], mode);
                ASSERT_EQ(d[lane], expected);
              }
              EXPECT_EQ(words[target][0], initial[target][0]);
              EXPECT_EQ(words[target][65], initial[target][65]);
            }
      }
}

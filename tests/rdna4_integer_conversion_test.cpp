// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_exec_masks.h"
#include "rdna4_integer_conversion_reference.h"

#include <algorithm>
#include <cfenv>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_rdna4_v_cvt_pk_i16_i32);
const Fn functions[] = {
    [](uint64_t f, uint64_t m, uint32_t i, uint32_t *const *d, const uint32_t *const *a,
       const uint32_t *const *) { return goc_rdna4_v_cvt_i32_i16(f, m, i, d, a); },
    [](uint64_t f, uint64_t m, uint32_t i, uint32_t *const *d, const uint32_t *const *a,
       const uint32_t *const *) { return goc_rdna4_v_cvt_u32_u16(f, m, i, d, a); },
    goc_rdna4_v_cvt_pk_i16_i32, goc_rdna4_v_cvt_pk_u16_u32};

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
        for (uint64_t mask : rdna4_exec_masks())
          for (unsigned breg = 0; breg < (op >= 2 ? 2u : 1u); ++breg)
            for (unsigned dreg = 0; dreg < 3; ++dreg) {
              uint32_t storage[3][34], expected[3][34];
              for (unsigned reg = 0; reg < 3; ++reg)
                for (unsigned word = 0; word < 34; ++word)
                  storage[reg][word] = expected[reg][word] = random();
              uint32_t mode = select ? GOC_ALU_HIGH_A : 0;
              for (unsigned lane = 0; lane < 32; ++lane)
                if ((mask >> lane) & 1)
                  expected[dreg][lane + 1] = goc_test::integer_conversion_reference(
                      op, storage[0][lane + 1], storage[breg][lane + 1], mode);
              const uint32_t *a[] = {storage[0] + 1}, *b[] = {storage[breg] + 1};
              uint32_t *d[] = {storage[dreg] + 1};
              ASSERT_EQ(functions[op](cpu | GOC_FP16_OVFL, mask, mode, d, a, b), GOC_SUCCESS);
              for (unsigned reg = 0; reg < 3; ++reg)
                for (unsigned word = 0; word < 34; ++word)
                  ASSERT_EQ(storage[reg][word], expected[reg][word]);
            }
}

TEST(IntegerConversion, HostEnvironmentIsPreserved) {
  fenv_t saved;
  ASSERT_EQ(std::fegetenv(&saved), 0);
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
  EXPECT_EQ(std::fesetenv(&saved), 0);
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
      EXPECT_EQ(functions[op](cpu | (UINT64_C(1) << 63), UINT32_MAX, 0, d, a, a),
                GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(functions[op](cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                              UINT32_MAX, 0, d, a, a),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
      for (auto word : output)
        EXPECT_EQ(word, 0xdeadbeefu);
      EXPECT_EQ(functions[op](cpu, UINT64_C(0xffffffff00000000), 0, nullptr, nullptr, nullptr),
                GOC_SUCCESS);
      for (unsigned sem = 0; sem < 4; ++sem) {
        EXPECT_EQ(functions[op](cpu | (uint64_t(sem) << 16), UINT32_MAX, 0, d, a, a), GOC_SUCCESS);
        for (auto word : output)
          EXPECT_EQ(word, 0u);
      }
    }
}

// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_exec_masks.h"
#include "rdna4_scalar_integer_hardware.h"
#include "rdna4_scalar_integer_reference.h"

#include <cfenv>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdint.h>

TEST(ScalarInteger, HardwareResultsAndScc) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
      for (unsigned seed = 0; seed < 2; ++seed)
        for (uint64_t mask : {UINT64_C(0), UINT64_MAX, UINT64_C(0xaaaaaaaa)})
          for (unsigned k = 0; k < 40; ++k) {
            uint64_t hash = UINT64_C(14695981039346656037);
            unsigned op = k < 18 ? k : k < 29 ? 18 : 19;
            uint16_t imm = k < 18 ? 0 : goc_test::scalar_integer_literals[(k - 18) % 11];
            for (unsigned i = 0; i < 4096; ++i) {
              uint32_t w[4];
              goc_test::scalar_integer_inputs(i, w);
              uint64_t a = (uint64_t(w[1]) << 32) | w[0], b = (uint64_t(w[3]) << 32) | w[2],
                       d64 = 0;
              uint32_t d = w[0], cc = seed;
              ASSERT_EQ(goc_test::scalar_integer_call(op, cpu | semantics | GOC_SEMANTICS_STRICT,
                                                      mask, 0, &d, &d64, a, b, &cc,
                                                      seed | 0xaaaa0000u, imm),
                        GOC_SUCCESS);
              bool wide = op >= 15 && op <= 17;
              for (uint32_t word : {wide ? uint32_t(d64) : d, wide ? uint32_t(d64 >> 32) : 0u, cc})
                hash = (hash ^ word) * UINT64_C(1099511628211);
            }
            ASSERT_EQ(hash, goc_test::scalar_integer_hardware[seed][k])
                << cpu << "/" << semantics << "/" << seed << "/" << mask << "/" << k;
          }
}

TEST(ScalarInteger, ExecAndAliasing) {
  for (unsigned op = 0; op < 20; ++op)
    for (uint64_t mask : rdna4_exec_masks()) {
      uint32_t d = 0x80000000u, cc = 0;
      uint64_t wide = 0;
      ASSERT_EQ(
          goc_test::scalar_integer_call(op, 0, mask, 0, &d, &wide, d, 0xffffffffu, &cc, 1, 0xffff),
          GOC_SUCCESS);
      uint32_t words[] = {0x12345678, 0x80000000u, 0x87654321};
      uint64_t wide2 = UINT64_C(0x80000000);
      ASSERT_EQ(goc_test::scalar_integer_call(op, 0, mask, 0, words + 1, &wide2, words[1],
                                              0xffffffffu, words + 1, 1, 0xffff),
                GOC_SUCCESS);
      ASSERT_EQ(words[1], op < 12 || op == 18 ? cc : op >= 15 && op <= 17 ? 0x80000000u : d);
      ASSERT_EQ(wide2, op >= 15 && op <= 17 ? wide : UINT64_C(0x80000000));
      ASSERT_EQ(words[0], 0x12345678u);
      ASSERT_EQ(words[2], 0x87654321u);
    }
}

TEST(ScalarInteger, AllSignedImmediates) {
  for (uint32_t a : {0u, 1u, 0x7fffffffu, 0x80000000u, 0xffffffffu})
    for (int imm = -32768; imm <= 32767; ++imm) {
      int64_t signed_a = a < 0x80000000u ? int64_t(a) : int64_t(a) - INT64_C(0x100000000);
      int64_t sum = signed_a + imm;
      uint32_t d = a, cc = 123;
      ASSERT_EQ(goc_rdna4_s_addk_co_i32(0, 0, 0, &d, uint16_t(imm), &cc), GOC_SUCCESS);
      ASSERT_EQ(d, uint32_t(sum));
      ASSERT_EQ(cc, unsigned(sum < -INT64_C(2147483648) || sum > INT64_C(2147483647)));
      d = a;
      ASSERT_EQ(goc_rdna4_s_mulk_i32(0, 0, 0, &d, uint16_t(imm)), GOC_SUCCESS);
      ASSERT_EQ(d, uint32_t(signed_a * imm));
    }
}

TEST(ScalarInteger, AbsDiffWrapsBeforeAbsoluteValue) {
  uint32_t d, cc;
  ASSERT_EQ(goc_rdna4_s_absdiff_i32(0, 0, 0, &d, 0x80000000u, 1, &cc), GOC_SUCCESS);
  EXPECT_EQ(d, 0x7fffffffu);
  EXPECT_EQ(cc, 1u);
  ASSERT_EQ(goc_rdna4_s_absdiff_i32(0, 0, 0, &d, 0x7fffffffu, 0xffffffffu, &cc), GOC_SUCCESS);
  EXPECT_EQ(d, 0x80000000u);
}

TEST(ScalarInteger, ErrorsDoNotWrite) {
  for (unsigned op = 0; op < 20; ++op) {
    uint32_t d = 123, cc = 456;
    uint64_t wide = 789;
    for (unsigned bit = 0; bit < 32; ++bit)
      EXPECT_EQ(goc_test::scalar_integer_call(op, 0, 0, 1u << bit, &d, &wide, 0, 0, &cc, 0, 0),
                GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(
        goc_test::scalar_integer_call(op, UINT64_C(1) << 63, 0, 0, &d, &wide, 0, 0, &cc, 0, 0),
        GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(d, 123u);
    EXPECT_EQ(cc, 456u);
    EXPECT_EQ(wide, 789u);
  }
}

TEST(ScalarInteger, HostFpStatePreserved) {
  fenv_t saved;
  ASSERT_EQ(std::fegetenv(&saved), 0);
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    std::fesetround(rounding);
    std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_DIVBYZERO);
    int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
    for (unsigned op = 0; op < 20; ++op) {
      uint32_t d = 0x80000000u, cc;
      uint64_t wide;
      EXPECT_EQ(goc_test::scalar_integer_call(op, 0, 0, 0, &d, &wide, 0xffffffffu, 0x80000000u, &cc,
                                              1, 0x8000),
                GOC_SUCCESS);
      EXPECT_EQ(std::fegetround(), rounding);
      EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
    }
  }
  EXPECT_EQ(std::fesetenv(&saved), 0);
}

TEST(ScalarInteger, SignExtendExhaustiveHardware) {
  // GFX1201 / HIP 7.13: every low 16-bit pattern with changing upper bits.
  // Full/empty/alternating EXEC and incoming SCC 0/1 yielded identical results
  // and preserved SCC: 786432 result/SCC triples checked against signed values.
  const uint64_t hashes[] = {UINT64_C(0x03ad8958c79f2325), UINT64_C(0x41d0f9b5b59f2325)};
  const auto functions = {goc_rdna4_s_sext_i32_i8, goc_rdna4_s_sext_i32_i16};
  unsigned op = 0;
  for (auto fn : functions) {
    unsigned modulus = op ? 65536 : 256;
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL}) {
        uint64_t hash = UINT64_C(14695981039346656037);
        for (unsigned i = 0; i < 65536; ++i) {
          uint32_t a = i | ((i * 0x9e37u) << 16), d;
          ASSERT_EQ(fn(cpu | semantics | GOC_SEMANTICS_STRICT, UINT64_MAX, 0, &d, a), GOC_SUCCESS);
          int value = int(i % modulus);
          if (value >= int(modulus / 2))
            value -= int(modulus);
          ASSERT_EQ(d, uint32_t(value));
          hash = (hash ^ d) * UINT64_C(1099511628211);
        }
        EXPECT_EQ(hash, hashes[op]);
      }
    ++op;
  }
}

TEST(ScalarInteger, SignExtendExecAliasesFlagsAndHostState) {
  fenv_t saved;
  ASSERT_EQ(std::fegetenv(&saved), 0);
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    ASSERT_EQ(std::fesetround(rounding), 0);
    std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_DIVBYZERO);
    int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
    for (auto fn : {goc_rdna4_s_sext_i32_i8, goc_rdna4_s_sext_i32_i16}) {
      for (uint64_t mask : rdna4_exec_masks()) {
        uint32_t words[] = {123, 0xabcdffff, 456};
        ASSERT_EQ(fn(0, mask, 0, words + 1, words[1]), GOC_SUCCESS);
        EXPECT_EQ(words[1], UINT32_MAX);
        EXPECT_EQ(words[0], 123u);
        EXPECT_EQ(words[2], 456u);
      }
      uint32_t d = 123;
      for (unsigned bit = 0; bit < 32; ++bit)
        EXPECT_EQ(fn(0, 0, 1u << bit, &d, 0), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(UINT64_C(1) << 63, 0, 0, &d, 0), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(d, 123u);
      EXPECT_EQ(std::fegetround(), rounding);
      EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
    }
  }
  EXPECT_EQ(std::fesetenv(&saved), 0);
}

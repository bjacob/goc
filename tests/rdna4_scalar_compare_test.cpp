// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "rdna4_scalar_compare_hardware.h"
#include "rdna4_scalar_compare_reference.h"
#include "rdna4_scalar_fp_reference.h"

#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdint.h>

TEST(ScalarCompare, HardwareEveryPredicateAndFpState) {
  for (unsigned state = 0; state < 8; ++state)
    for (unsigned op = 0; op < 46; ++op)
      for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
          uint64_t hash = goc_test::capture_hash_seed;
          for (unsigned i = 0; i < 4096; ++i) {
            uint32_t w[4], scc = 123;
            goc_test::scalar_compare_inputs(i, op, w);
            uint64_t a = (uint64_t(w[1]) << 32) | w[0], b = (uint64_t(w[3]) << 32) | w[2];
            ASSERT_EQ(goc_test::scalar_compare_call(op,
                                                    cpu | semantics | GOC_SEMANTICS_STRICT |
                                                        goc_test::scalar_fp_flags(state),
                                                    0, &scc, a, b),
                      GOC_SUCCESS);
            hash = goc_test::capture_hash_word(hash, scc);
          }
          ASSERT_EQ(hash, goc_test::scalar_compare_hardware[state][op])
              << state << "/" << op << "/" << semantics << "/" << cpu;
        }
}

TEST(ScalarCompare, OutputMayOverlapEitherSourceWord) {
  for (unsigned op = 0; op < 46; ++op)
    for (unsigned sample : {0u, 3u, 33u, 127u, 555u, 3001u}) {
      uint32_t w[4], expected;
      goc_test::scalar_compare_inputs(sample, op, w);
      uint64_t a = (uint64_t(w[1]) << 32) | w[0], b = (uint64_t(w[3]) << 32) | w[2];
      ASSERT_EQ(goc_test::scalar_compare_call(op, 0, 0, &expected, a, b), GOC_SUCCESS);

      for (unsigned alias = 0; alias < 4; ++alias) {
        uint64_t words[] = {123, a, b, 456}, want[] = {123, a, b, 456};
        auto out =
            reinterpret_cast<uint32_t *>(reinterpret_cast<unsigned char *>(words + 1) + alias * 4);
        std::memcpy(reinterpret_cast<unsigned char *>(want + 1) + alias * 4, &expected, 4);
        ASSERT_EQ(goc_test::scalar_compare_call(op, 0, 0, out, words[1], words[2]), GOC_SUCCESS);
        for (unsigned j = 0; j < 4; ++j)
          EXPECT_EQ(words[j], want[j]);
      }
    }
}

TEST(ScalarCompare, EveryBitIndexAndModuloCount) {
  for (unsigned op = 12; op < 16; ++op) {
    unsigned width = op < 14 ? 32 : 64, wanted = op & 1;
    for (unsigned bit = 0; bit < width; ++bit)
      for (unsigned index = 0; index < 256; ++index) {
        uint32_t scc;
        ASSERT_EQ(goc_test::scalar_compare_call(op, 0, 0, &scc, 1ULL << bit, 0xffff0000u | index),
                  GOC_SUCCESS);
        EXPECT_EQ(scc, unsigned(unsigned(bit == index % width) == wanted));
      }
  }
}

TEST(ScalarCompare, NaNsZerosDenormalsAndUnsigned64) {
  uint32_t scc;
  ASSERT_EQ(goc_rdna4_s_cmp_lt_f32(0, 0, &scc, 0x80000001, 0, nullptr), GOC_SUCCESS);
  EXPECT_EQ(scc, 1u);
  ASSERT_EQ(goc_rdna4_s_cmp_lt_f32(GOC_FP_FLUSH_INPUT_DENORMALS, 0, &scc, 0x80000001, 0, nullptr),
            GOC_SUCCESS);
  EXPECT_EQ(scc, 0u);
  ASSERT_EQ(goc_rdna4_s_cmp_eq_f16(0, 0, &scc, 0xabcd8000, 0x12340000, nullptr), GOC_SUCCESS);
  EXPECT_EQ(scc, 1u);
  ASSERT_EQ(goc_rdna4_s_cmp_lg_f32(0, 0, &scc, 0x7f800001, 0, nullptr), GOC_SUCCESS);
  EXPECT_EQ(scc, 0u);
  ASSERT_EQ(goc_rdna4_s_cmp_neq_f32(0, 0, &scc, 0x7f800001, 0, nullptr), GOC_SUCCESS);
  EXPECT_EQ(scc, 1u);
  ASSERT_EQ(goc_rdna4_s_cmp_nlg_f32(0, 0, &scc, 0x7f800001, 0, nullptr), GOC_SUCCESS);
  EXPECT_EQ(scc, 1u);
  ASSERT_EQ(goc_rdna4_s_cmp_eq_u64(0, 0, &scc, 0x100000000ULL, 0), GOC_SUCCESS);
  EXPECT_EQ(scc, 0u);
  ASSERT_EQ(goc_rdna4_s_cmp_lg_u64(0, 0, &scc, UINT64_MAX, UINT64_MAX), GOC_SUCCESS);
  EXPECT_EQ(scc, 0u);
}

TEST(ScalarCompare, ErrorsAndHostFpEnvironment) {
  goc_test::ScopedFpEnvironment restore;
  ASSERT_TRUE(restore.saved());
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    ASSERT_EQ(std::fesetround(rounding), 0);
    std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_DIVBYZERO);
    int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
    for (unsigned op = 0; op < 46; ++op) {
      uint32_t scc = 123;
      for (unsigned bit = 0; bit < 32; ++bit)
        EXPECT_EQ(goc_test::scalar_compare_call(op, 0, 1u << bit, &scc, 0, 0),
                  GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(goc_test::scalar_compare_call(op, 1ULL << 63, 0, &scc, 0, 0),
                GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(scc, 123u);
      EXPECT_EQ(goc_test::scalar_compare_call(op, GOC_SEMANTICS_EXACT_EMPIRICAL, 0, &scc,
                                              0x7f807c01u, 0xff807c01u),
                GOC_SUCCESS);
      EXPECT_EQ(std::fegetround(), rounding);
      EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
    }
  }
}

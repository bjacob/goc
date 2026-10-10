// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "scalar_convert_hardware.h"
#include "scalar_convert_reference.h"
#include "scalar_fp_reference.h"

#include <cfenv>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdint.h>

TEST(ScalarConvert, HardwareAllHalfPatternsAndFpStates) {
  for (unsigned state = 0; state < 8; ++state)
    for (unsigned op = 0; op < 8; ++op)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
        uint64_t hash = goc_test::capture_hash_seed;
        for (unsigned i = 0; i < 65536; ++i) {
          uint32_t w[2], d;
          goc_test::scalar_convert_inputs(i, op, w);
          ASSERT_EQ(goc_test::scalar_convert_call(op, cpu | goc_test::scalar_fp_flags(state), 0, &d,
                                                  w[0], w[1]),
                    GOC_SUCCESS);
          hash = goc_test::capture_hash_word(hash, d);
        }
        ASSERT_EQ(hash, goc_test::scalar_convert_hardware[state][op])
            << state << "/" << op << "/" << cpu;
      }
}

TEST(ScalarConvert, Aliasing) {
  for (unsigned state = 0; state < 8; ++state)
    for (unsigned op = 0; op < 8; ++op)
      for (unsigned i : {0u, 1u, 8u, 16u, 22u, 26u, 30u, 0x7c01u, 0x8001u, 0xffffu}) {
        uint32_t w[2], expected;
        goc_test::scalar_convert_inputs(i, op, w);
        uint64_t flags = goc_test::scalar_fp_flags(state);
        ASSERT_EQ(goc_test::scalar_convert_call(op, flags, 0, &expected, w[0], w[1]), GOC_SUCCESS);

        for (unsigned alias = 1; alias <= 2; ++alias) {
          uint32_t words[] = {123, w[0], w[1], 456};
          ASSERT_EQ(goc_test::scalar_convert_call(op, flags, 0, words + alias, words[1], words[2]),
                    GOC_SUCCESS);
          EXPECT_EQ(words[alias], expected);
          EXPECT_EQ(words[0], 123u);
          EXPECT_EQ(words[3], 456u);
          EXPECT_EQ(words[3 - alias], w[2 - alias]);
        }
      }
}

TEST(ScalarConvert, SaturationNaNsRoundingAndHalfSelection) {
  uint32_t d;
  ASSERT_EQ(goc_s_cvt_f32_i32(0, 0, &d, 0xffffffff, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0xbf800000u);
  ASSERT_EQ(goc_s_cvt_f32_u32(0, 0, &d, 0x01000001, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0x4b800000u);
  ASSERT_EQ(goc_s_cvt_f32_u32(0, 0, &d, 0x01000003, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0x4b800002u);
  ASSERT_EQ(goc_s_cvt_i32_f32(0, 0, &d, 0x7f800000, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0x7fffffffu);
  ASSERT_EQ(goc_s_cvt_i32_f32(0, 0, &d, 0xff800000, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0x80000000u);
  ASSERT_EQ(goc_s_cvt_i32_f32(0, 0, &d, 0xff800001, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0u);
  ASSERT_EQ(goc_s_cvt_u32_f32(0, 0, &d, 0xbf800000, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0u);
  ASSERT_EQ(goc_s_cvt_u32_f32(0, 0, &d, 0x4f800000, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, UINT32_MAX);
  ASSERT_EQ(goc_s_cvt_f32_f16(0, 0, &d, 0x3c00bc00, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0xbf800000u);
  ASSERT_EQ(goc_s_cvt_hi_f32_f16(0, 0, &d, 0x3c00bc00, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0x3f800000u);
  ASSERT_EQ(goc_s_cvt_f16_f32(0, 0, &d, 0x7f800001, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0x7e00u);
  ASSERT_EQ(goc_s_cvt_f16_f32(GOC_FP16_OVFL, 0, &d, 0x47800000, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0x7bffu);
  ASSERT_EQ(goc_s_cvt_pk_rtz_f16_f32(0, 0, &d, 0x47800000, 0xff800000, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0xfc007bffu);
  ASSERT_EQ(goc_s_cvt_f16_f32(GOC_FP_FLUSH_OUTPUT_DENORMALS, 0, &d, 0x387fe000, nullptr),
            GOC_SUCCESS);
  EXPECT_EQ(d, 0u);
  ASSERT_EQ(goc_s_cvt_f16_f32(0, 0, &d, 0x387fe000, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0x400u);
  ASSERT_EQ(goc_s_cvt_f32_f16(GOC_FP_FLUSH_INPUT_DENORMALS, 0, &d, 0x8001, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0x80000000u);
}

TEST(ScalarConvert, ErrorsAndHostRounding) {
  goc_test::ScopedFpEnvironment restore;
  ASSERT_TRUE(restore.saved());
  ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
  for (unsigned op = 0; op < 8; ++op) {
    uint32_t d = 123;
    for (unsigned bit = 0; bit < 32; ++bit)
      EXPECT_EQ(goc_test::scalar_convert_call(op, 0, 1u << bit, &d, 0, 0), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_test::scalar_convert_call(op, 1ULL << 63, 0, &d, 0, 0), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_test::scalar_convert_call(
                  op, GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, &d, 0, 0),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    EXPECT_EQ(d, 123u);
    EXPECT_EQ(goc_test::scalar_convert_call(op, 0, 0, &d, 0x3fc00000, 0), GOC_SUCCESS);
    EXPECT_EQ(std::fegetround(), FE_TONEAREST);
  }
}

TEST(ScalarConvert, TruncationBoundariesAcrossScalarAndVector) {
  // Input bits, signed result, unsigned result. Include the neighbors on both
  // sides of the signed boundary and below the unsigned saturation boundary.
  const uint32_t cases[][3] = {{0x4effffff, 0x7fffff80, 0x7fffff80},
                               {0x4f000000, 0x7fffffff, 0x80000000},
                               {0x4f7fffff, 0x7fffffff, 0xffffff00},
                               {0x4f800000, 0x7fffffff, 0xffffffff},
                               {0xceffffff, 0x80000080, 0},
                               {0xcf000000, 0x80000000, 0},
                               {0x7fc00001, 0, 0},
                               {0xffc00001, 0, 0}};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (const auto &sample : cases) {
      uint32_t scalar, input[32], output[32];
      for (uint32_t &word : input)
        word = sample[0];
      const uint32_t *a = input;
      uint32_t *d = output;
      ASSERT_EQ(goc_s_cvt_i32_f32(cpu, 0, &scalar, sample[0], nullptr), GOC_SUCCESS);
      EXPECT_EQ(scalar, sample[1]);
      ASSERT_EQ(goc_v_cvt_i32_f32_rdna4(cpu, UINT32_MAX, 0, &d, &a, nullptr), GOC_SUCCESS);
      for (uint32_t word : output)
        EXPECT_EQ(word, sample[1]);
      ASSERT_EQ(goc_s_cvt_u32_f32(cpu, 0, &scalar, sample[0], nullptr), GOC_SUCCESS);
      EXPECT_EQ(scalar, sample[2]);
      ASSERT_EQ(goc_v_cvt_u32_f32_rdna4(cpu, UINT32_MAX, 0, &d, &a, nullptr), GOC_SUCCESS);
      for (uint32_t word : output)
        EXPECT_EQ(word, sample[2]);
    }
}

TEST(ScalarConvert, OptionalStateAndLooseFallbackContracts) {
  const decltype(&goc_s_cvt_f32_i32) unary[] = {
      goc_s_cvt_f32_i32, goc_s_cvt_f32_u32, goc_s_cvt_i32_f32,   goc_s_cvt_u32_f32,
      goc_s_cvt_f16_f32, goc_s_cvt_f32_f16, goc_s_cvt_hi_f32_f16};
  for (unsigned op = 0; op < 8; ++op) {
    auto call = [&](uint64_t flags, uint64_t mode, uint32_t *d, uint32_t a, uint32_t *state) {
      return op == 7 ? goc_s_cvt_pk_rtz_f16_f32(flags, mode, d, a, 0x80000001, state)
                     : unary[op](flags, mode, d, a, state);
    };
    for (unsigned state = 0; state < 8; ++state)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (uint32_t a : {0U, 0x80000000U, 0x7f800001U, 0x387fe000U, 0x3fc00000U}) {
          uint64_t flags = cpu | goc_test::scalar_fp_flags(state);
          uint32_t result, fallback, exceptions = 0xa5000040;
          ASSERT_EQ(call(flags, 0, &result, a, &exceptions), GOC_SUCCESS);
          EXPECT_EQ(exceptions, 0xa5000040U);
          ASSERT_EQ(call(flags | GOC_SEMANTICS_EXACT_EMPIRICAL, 0, &fallback, a, nullptr),
                    GOC_SUCCESS);
          EXPECT_EQ(fallback, result);
        }
    for (uint64_t semantics : {GOC_SEMANTICS_EXACT_EMPIRICAL, 2 * GOC_SEMANTICS_EXACT_EMPIRICAL,
                               3 * GOC_SEMANTICS_EXACT_EMPIRICAL})
      for (uint64_t strict : {uint64_t(0), GOC_SEMANTICS_STRICT})
        for (uint64_t mode : {uint64_t(0), uint64_t(GOC_ALU_NEG_A), uint64_t(1) << 63}) {
          uint32_t output = 0xdeadbeef, exceptions = 0xa5000040;
          EXPECT_EQ(call(semantics | strict, mode, &output, 0x7f800001, &exceptions),
                    GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
          EXPECT_EQ(output, 0xdeadbeefU);
          EXPECT_EQ(exceptions, 0xa5000040U);
        }
    for (unsigned bit = 0; bit < 64; ++bit) {
      uint32_t output = 0xdeadbeef, exceptions = 0xa5000040;
      EXPECT_EQ(call(GOC_SEMANTICS_LOOSE, 1ULL << bit, &output, 0, &exceptions),
                GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(output, 0xdeadbeefU);
      EXPECT_EQ(exceptions, 0xa5000040U);
    }
  }
}

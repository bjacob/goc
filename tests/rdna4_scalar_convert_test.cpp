// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_exec_masks.h"
#include "rdna4_scalar_convert_hardware.h"
#include "rdna4_scalar_convert_reference.h"
#include "rdna4_scalar_fp_reference.h"

#include <cfenv>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdint.h>

TEST(ScalarConvert, HardwareAllHalfPatternsAndFpStates) {
  for (unsigned state = 0; state < 8; ++state)
    for (unsigned op = 0; op < 8; ++op)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
        uint64_t hash = UINT64_C(14695981039346656037);
        for (unsigned i = 0; i < 65536; ++i) {
          uint32_t w[2], d;
          goc_test::scalar_convert_inputs(i, op, w);
          ASSERT_EQ(goc_test::scalar_convert_call(op, cpu | goc_test::scalar_fp_flags(state),
                                                  UINT64_MAX, 0, &d, w[0], w[1]),
                    GOC_SUCCESS);
          hash = (hash ^ d) * UINT64_C(1099511628211);
        }
        ASSERT_EQ(hash, goc_test::scalar_convert_hardware[state][op])
            << state << "/" << op << "/" << cpu;
      }
}

TEST(ScalarConvert, ExecAndAliasing) {
  for (unsigned state = 0; state < 8; ++state)
    for (unsigned op = 0; op < 8; ++op)
      for (unsigned i : {0u, 1u, 8u, 16u, 22u, 26u, 30u, 0x7c01u, 0x8001u, 0xffffu}) {
        uint32_t w[2], expected;
        goc_test::scalar_convert_inputs(i, op, w);
        uint64_t flags = goc_test::scalar_fp_flags(state);
        ASSERT_EQ(goc_test::scalar_convert_call(op, flags, UINT64_MAX, 0, &expected, w[0], w[1]),
                  GOC_SUCCESS);
        for (uint64_t mask : rdna4_exec_masks())
          for (unsigned alias = 1; alias <= 2; ++alias) {
            uint32_t words[] = {123, w[0], w[1], 456};
            ASSERT_EQ(goc_test::scalar_convert_call(op, flags, mask, 0, words + alias, words[1],
                                                    words[2]),
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
  ASSERT_EQ(goc_rdna4_s_cvt_f32_i32(0, 0, 0, &d, 0xffffffff), GOC_SUCCESS);
  EXPECT_EQ(d, 0xbf800000u);
  ASSERT_EQ(goc_rdna4_s_cvt_f32_u32(0, 0, 0, &d, 0x01000001), GOC_SUCCESS);
  EXPECT_EQ(d, 0x4b800000u);
  ASSERT_EQ(goc_rdna4_s_cvt_f32_u32(0, 0, 0, &d, 0x01000003), GOC_SUCCESS);
  EXPECT_EQ(d, 0x4b800002u);
  ASSERT_EQ(goc_rdna4_s_cvt_i32_f32(0, 0, 0, &d, 0x7f800000), GOC_SUCCESS);
  EXPECT_EQ(d, 0x7fffffffu);
  ASSERT_EQ(goc_rdna4_s_cvt_i32_f32(0, 0, 0, &d, 0xff800000), GOC_SUCCESS);
  EXPECT_EQ(d, 0x80000000u);
  ASSERT_EQ(goc_rdna4_s_cvt_i32_f32(0, 0, 0, &d, 0xff800001), GOC_SUCCESS);
  EXPECT_EQ(d, 0u);
  ASSERT_EQ(goc_rdna4_s_cvt_u32_f32(0, 0, 0, &d, 0xbf800000), GOC_SUCCESS);
  EXPECT_EQ(d, 0u);
  ASSERT_EQ(goc_rdna4_s_cvt_u32_f32(0, 0, 0, &d, 0x4f800000), GOC_SUCCESS);
  EXPECT_EQ(d, UINT32_MAX);
  ASSERT_EQ(goc_rdna4_s_cvt_f32_f16(0, 0, 0, &d, 0x3c00bc00), GOC_SUCCESS);
  EXPECT_EQ(d, 0xbf800000u);
  ASSERT_EQ(goc_rdna4_s_cvt_hi_f32_f16(0, 0, 0, &d, 0x3c00bc00), GOC_SUCCESS);
  EXPECT_EQ(d, 0x3f800000u);
  ASSERT_EQ(goc_rdna4_s_cvt_f16_f32(0, 0, 0, &d, 0x7f800001), GOC_SUCCESS);
  EXPECT_EQ(d, 0x7e00u);
  ASSERT_EQ(goc_rdna4_s_cvt_f16_f32(GOC_FP16_OVFL, 0, 0, &d, 0x47800000), GOC_SUCCESS);
  EXPECT_EQ(d, 0x7bffu);
  ASSERT_EQ(goc_rdna4_s_cvt_pk_rtz_f16_f32(0, 0, 0, &d, 0x47800000, 0xff800000), GOC_SUCCESS);
  EXPECT_EQ(d, 0xfc007bffu);
  ASSERT_EQ(goc_rdna4_s_cvt_f16_f32(GOC_FP_FLUSH_OUTPUT_DENORMALS, 0, 0, &d, 0x387fe000),
            GOC_SUCCESS);
  EXPECT_EQ(d, 0u);
  ASSERT_EQ(goc_rdna4_s_cvt_f16_f32(0, 0, 0, &d, 0x387fe000), GOC_SUCCESS);
  EXPECT_EQ(d, 0x400u);
  ASSERT_EQ(goc_rdna4_s_cvt_f32_f16(GOC_FP_FLUSH_INPUT_DENORMALS, 0, 0, &d, 0x8001), GOC_SUCCESS);
  EXPECT_EQ(d, 0x80000000u);
}

TEST(ScalarConvert, ErrorsAndHostRounding) {
  fenv_t saved;
  ASSERT_EQ(std::fegetenv(&saved), 0);
  ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
  for (unsigned op = 0; op < 8; ++op) {
    uint32_t d = 123;
    for (unsigned bit = 0; bit < 32; ++bit)
      EXPECT_EQ(goc_test::scalar_convert_call(op, 0, 0, 1u << bit, &d, 0, 0),
                GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_test::scalar_convert_call(op, UINT64_C(1) << 63, 0, 0, &d, 0, 0),
              GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_test::scalar_convert_call(
                  op, GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0, &d, 0, 0),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    EXPECT_EQ(d, 123u);
    EXPECT_EQ(goc_test::scalar_convert_call(op, 0, 0, 0, &d, 0x3fc00000, 0), GOC_SUCCESS);
    EXPECT_EQ(std::fegetround(), FE_TONEAREST);
  }
  EXPECT_EQ(std::fesetenv(&saved), 0);
}

// SPDX-License-Identifier: MIT

#include "fp_environment.h"
#include "goc/goc.h"
#include "rdna4_exec_masks.h"
#include "rdna4_scalar_fma_hardware.h"
#include "rdna4_scalar_fma_reference.h"
#include "rdna4_scalar_fp_reference.h"

#include <cfenv>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdint.h>

TEST(ScalarFma, HardwareEveryFpStateAndLiteral) {
  for (unsigned state = 0; state < 8; ++state)
    for (unsigned k = 0; k < 26; ++k)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
        uint64_t hash = UINT64_C(14695981039346656037);
        for (unsigned i = 0; i < 4096; ++i) {
          uint32_t w[3];
          goc_test::scalar_fma_inputs(i, k == 1, w);
          uint32_t d = w[2];
          ASSERT_EQ(goc_test::scalar_fma_call(k, cpu | goc_test::scalar_fp_flags(state), UINT32_MAX,
                                              0, &d, w[0], w[1]),
                    GOC_SUCCESS);
          hash = (hash ^ goc_test::scalar_fp_canonical(d, k == 1)) * UINT64_C(1099511628211);
        }
        ASSERT_EQ(hash, goc_test::scalar_fma_hardware[state][k]) << state << "/" << k << "/" << cpu;
      }
}

TEST(ScalarFma, ExecAndAllScalarAliases) {
  for (unsigned state = 0; state < 8; ++state)
    for (unsigned k = 0; k < 26; ++k)
      for (unsigned sample : {0u, 6u, 18u, 1023u, 3001u}) {
        uint32_t w[3];
        goc_test::scalar_fma_inputs(sample, k == 1, w);
        auto flags = goc_test::scalar_fp_flags(state);
        for (unsigned alias = 0; alias < 3; ++alias) {
          uint32_t expected = w[alias];
          ASSERT_EQ(goc_test::scalar_fma_call(k, flags, UINT32_MAX, 0, &expected, w[0], w[1]),
                    GOC_SUCCESS);
          for (uint32_t mask : rdna4_exec_masks()) {
            uint32_t words[] = {123, w[0], w[1], w[2], 456};
            ASSERT_EQ(
                goc_test::scalar_fma_call(k, flags, mask, 0, words + 1 + alias, words[1], words[2]),
                GOC_SUCCESS);
            EXPECT_EQ(goc_test::scalar_fp_canonical(words[1 + alias], k == 1),
                      goc_test::scalar_fp_canonical(expected, k == 1));
            for (unsigned j = 0; j < 3; ++j)
              if (j != alias) {
                EXPECT_EQ(words[j + 1], w[j]);
              }
            EXPECT_EQ(words[0], 123u);
            EXPECT_EQ(words[4], 456u);
          }
        }
      }
}

TEST(ScalarFma, TrueFusionAndGuestFpStages) {
  uint32_t d = 0xbf800000;
  ASSERT_EQ(goc_rdna4_s_fmac_f32(0, 0, 0, &d, 0x3f800001, 0x3f7ffffe), GOC_SUCCESS);
  EXPECT_EQ(d, 0xa8800000u);
  d = 0xabcdbc00;
  ASSERT_EQ(goc_rdna4_s_fmac_f16(0, 0, 0, &d, 0x3c01, 0x3bfe), GOC_SUCCESS);
  EXPECT_EQ(d, 0x8010u);
  d = 0;
  ASSERT_EQ(goc_rdna4_s_fmac_f32(GOC_FP_FLUSH_OUTPUT_DENORMALS, 0, 0, &d, 0x00800000, 0x3f7fffff),
            GOC_SUCCESS);
  EXPECT_EQ(d, 0u);
  d = 0;
  ASSERT_EQ(goc_rdna4_s_fmac_f32(0, 0, 0, &d, 0x00800000, 0x3f7fffff), GOC_SUCCESS);
  EXPECT_EQ(d, 0x00800000u);
  d = 1;
  ASSERT_EQ(goc_rdna4_s_fmac_f16(GOC_FP_FLUSH_INPUT_DENORMALS, 0, 0, &d, 0x3c00, 0), GOC_SUCCESS);
  EXPECT_EQ(d, 0u);
  d = 0;
  ASSERT_EQ(goc_rdna4_s_fmac_f16(GOC_FP16_OVFL, 0, 0, &d, 0x7bff, 0x4000), GOC_SUCCESS);
  EXPECT_EQ(d, 0x7bffu);
  ASSERT_EQ(goc_rdna4_s_fmaak_f32(0, 0, 0, &d, 0x3f800001, 0x3f7ffffe, 0xbf800000), GOC_SUCCESS);
  EXPECT_EQ(d, 0xa8800000u);
  ASSERT_EQ(goc_rdna4_s_fmamk_f32(0, 0, 0, &d, 0x3f800001, 0x3f7ffffe, 0xbf800000), GOC_SUCCESS);
  EXPECT_EQ(d, 0xa8800000u);
}

TEST(ScalarFma, ErrorsAndHostRounding) {
  goc_test::ScopedFpEnvironment restore;
  ASSERT_TRUE(restore.saved());
  ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
  for (unsigned k = 0; k < 26; ++k) {
    uint32_t d = 123;
    for (unsigned bit = 0; bit < 32; ++bit)
      EXPECT_EQ(goc_test::scalar_fma_call(k, 0, 0, 1u << bit, &d, 0, 0), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_test::scalar_fma_call(k, UINT64_C(1) << 63, 0, 0, &d, 0, 0),
              GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_test::scalar_fma_call(k, GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0,
                                        0, &d, 0, 0),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    EXPECT_EQ(d, 123u);
    EXPECT_EQ(goc_test::scalar_fma_call(k, 0, 0, 0, &d, 0, 0), GOC_SUCCESS);
    EXPECT_EQ(std::fegetround(), FE_TONEAREST);
  }
}

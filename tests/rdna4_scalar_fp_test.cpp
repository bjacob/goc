// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "rdna4_exec_masks.h"
#include "rdna4_scalar_fp_hardware.h"
#include "rdna4_scalar_fp_reference.h"

#include <cfenv>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdint.h>

TEST(ScalarFp, HardwareEveryFpStateAndUnderflowBoundary) {
  for (unsigned dataset = 0; dataset < 2; ++dataset)
    for (unsigned state = 0; state < 8; ++state)
      for (unsigned op = 0; op < 14; ++op)
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
          uint64_t hash = goc_test::capture_hash_seed;
          for (unsigned i = 0; i < 4096; ++i) {
            uint32_t w[2], d;
            if (dataset)
              goc_test::scalar_fp_boundary_inputs(i, op & 1, w);
            else
              goc_test::scalar_fp_inputs(i, op & 1, w);
            ASSERT_EQ(goc_test::scalar_fp_functions[op](cpu | goc_test::scalar_fp_flags(state),
                                                        UINT32_MAX, 0, &d, w[0], w[1], nullptr),
                      GOC_SUCCESS);
            hash = goc_test::capture_hash_word(hash, goc_test::scalar_fp_canonical(d, op & 1));
          }
          ASSERT_EQ(hash, goc_test::scalar_fp_hardware[dataset][state][op])
              << dataset << "/" << state << "/" << op << "/" << cpu;
        }
}

TEST(ScalarFp, ExecIgnoredAndInputStorageMayAlias) {
  for (unsigned state = 0; state < 8; ++state)
    for (unsigned op = 0; op < 14; ++op)
      for (unsigned sample : {0u, 3u, 216u, 774u, 1023u, 3001u}) {
        uint32_t w[2], expected;
        goc_test::scalar_fp_inputs(sample, op & 1, w);
        auto flags = goc_test::scalar_fp_flags(state);
        ASSERT_EQ(
            goc_test::scalar_fp_functions[op](flags, UINT32_MAX, 0, &expected, w[0], w[1], nullptr),
            GOC_SUCCESS);
        for (uint32_t exec_mask : rdna4_exec_masks()) {
          uint32_t storage[] = {123, w[0], w[1], 456};
          ASSERT_EQ(goc_test::scalar_fp_functions[op](flags, exec_mask, 0, storage + 1, storage[1],
                                                      storage[2], nullptr),
                    GOC_SUCCESS);
          EXPECT_EQ(goc_test::scalar_fp_canonical(storage[1], op & 1),
                    goc_test::scalar_fp_canonical(expected, op & 1));
          EXPECT_EQ(storage[0], 123u);
          EXPECT_EQ(storage[2], w[1]);
          EXPECT_EQ(storage[3], 456u);
        }
      }
}

TEST(ScalarFp, DenormalStagesAndFiniteOverflow) {
  uint32_t d;
  const uint64_t output = GOC_FP_FLUSH_OUTPUT_DENORMALS;
  ASSERT_EQ(goc_rdna4_s_mul_f32(output, 0, 0, &d, 0x00800000, 0x3f7fffff, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0u);
  ASSERT_EQ(goc_rdna4_s_mul_f32(0, 0, 0, &d, 0x00800000, 0x3f7fffff, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0x00800000u);
  ASSERT_EQ(goc_rdna4_s_mul_f32(output, 0, 0, &d, 0x007fffff, 0x3f800001, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0x00800000u);
  ASSERT_EQ(goc_rdna4_s_mul_f16(output, 0, 0, &d, 0x0400, 0x3bff, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0u);
  ASSERT_EQ(goc_rdna4_s_mul_f16(output, 0, 0, &d, 0x03ff, 0x3c01, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0x0400u);
  ASSERT_EQ(goc_rdna4_s_min_num_f32(output, 0, 0, &d, 0x80000001, 0, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0x80000001u);
  ASSERT_EQ(goc_rdna4_s_min_num_f32(GOC_FP_FLUSH_INPUT_DENORMALS, 0, 0, &d, 0x80000001, 0, nullptr),
            GOC_SUCCESS);
  EXPECT_EQ(d, 0x80000000u);
  ASSERT_EQ(goc_rdna4_s_mul_f16(GOC_FP16_OVFL, 0, 0, &d, 0x7bff, 0x4000, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0x7bffu);
  ASSERT_EQ(goc_rdna4_s_mul_f16(GOC_FP16_OVFL, 0, 0, &d, 0x7c00, 0x4000, nullptr), GOC_SUCCESS);
  EXPECT_EQ(d, 0x7c00u);
}

TEST(ScalarFp, ErrorsLeaveOutputUntouchedAndRoundingPreserved) {
  goc_test::ScopedFpEnvironment restore;
  ASSERT_TRUE(restore.saved());
  ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
  for (unsigned op = 0; op < 14; ++op) {
    uint32_t d = 123;
    for (unsigned bit = 0; bit < 32; ++bit)
      EXPECT_EQ(goc_test::scalar_fp_functions[op](0, 0, 1u << bit, &d, 0, 0, nullptr),
                GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_test::scalar_fp_functions[op](1ULL << 63, 0, 0, &d, 0, 0, nullptr),
              GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_test::scalar_fp_functions[op](
                  GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0, &d, 0, 0, nullptr),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    EXPECT_EQ(d, 123u);
    EXPECT_EQ(goc_test::scalar_fp_functions[op](0, 0, 0, &d, 0, 0, nullptr), GOC_SUCCESS);
    EXPECT_EQ(std::fegetround(), FE_TONEAREST);
  }
}

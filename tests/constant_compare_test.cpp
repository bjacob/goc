// SPDX-License-Identifier: MIT

#include "dpp_reference.h"
#include "goc/goc.h"

#include <gtest/gtest.h>
#include <stdint.h>

TEST(ConstantCompare, PredicatesMasksAndDppFiltering) {
  const decltype(&goc_v_cmp_f_i32) functions[] = {
      goc_v_cmp_f_i32, goc_v_cmp_t_i32, goc_v_cmpx_f_i32, goc_v_cmpx_t_i32,
      goc_v_cmp_f_u32, goc_v_cmp_t_u32, goc_v_cmpx_f_u32, goc_v_cmpx_t_u32,
      goc_v_cmp_f_i64, goc_v_cmp_t_i64, goc_v_cmpx_f_i64, goc_v_cmpx_t_i64,
      goc_v_cmp_f_u64, goc_v_cmp_t_u64, goc_v_cmpx_f_u64, goc_v_cmpx_t_u64};
  for (unsigned op = 0; op < 16; ++op)
    for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (uint32_t exec_mask : {0U, UINT32_MAX, 0xaaaaaaaaU, 0x80000001U, 0x00ff00ffU}) {
          uint32_t d = 123;
          ASSERT_EQ(functions[op](cpu | semantics, exec_mask, 0, &d, nullptr, nullptr),
                    GOC_SUCCESS);
          EXPECT_EQ(d, (op & 1) ? exec_mask : 0);
          for (auto mode : goc_test::dpp_modes) {
            d = 123;
            if (op >= 8) {
              EXPECT_EQ(functions[op](cpu | semantics, exec_mask, mode, &d, nullptr, nullptr),
                        GOC_ERROR_INVALID_FLAGS);
              EXPECT_EQ(d, 123U);
            } else {
              uint32_t expected = 0;
              for (unsigned lane = 0; lane < 32; ++lane) {
                int source;
                if ((op & 1) && goc_test::dpp_source(mode, exec_mask, lane, source))
                  expected |= 1U << lane;
              }
              ASSERT_EQ(functions[op](cpu | semantics, exec_mask, mode, &d, nullptr, nullptr),
                        GOC_SUCCESS);
              EXPECT_EQ(d, expected);
            }
          }
          for (unsigned bit = 0; bit < 32; ++bit) {
            d = 123;
            EXPECT_EQ(functions[op](cpu | semantics, exec_mask, 1ULL << bit, &d, nullptr, nullptr),
                      GOC_ERROR_INVALID_FLAGS);
            EXPECT_EQ(d, 123U);
          }
        }
}

namespace {

const decltype(&goc_v_cmp_f_f16) floating[] = {
    goc_v_cmp_f_f16, goc_v_cmp_t_f16, goc_v_cmpx_f_f16, goc_v_cmpx_t_f16,
    goc_v_cmp_f_f32, goc_v_cmp_t_f32, goc_v_cmpx_f_f32, goc_v_cmpx_t_f32,
    goc_v_cmp_f_f64, goc_v_cmp_t_f64, goc_v_cmpx_f_f64, goc_v_cmpx_t_f64};

} // namespace

TEST(ConstantCompare, FloatingResultsMasksAndDpp) {
  for (unsigned op = 0; op < 12; ++op)
    for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (uint32_t exec_mask : {0U, 1U, 0x80000000U, 0xaaaaaaaaU, UINT32_MAX}) {
          uint64_t flags = cpu | semantics | GOC_SEMANTICS_STRICT | GOC_FP_FLUSH_INPUT_DENORMALS |
                           GOC_FP_FLUSH_OUTPUT_DENORMALS | GOC_FP_IEEE_MODE;
          uint32_t d = 123;
          ASSERT_EQ(floating[op](flags, exec_mask, 0, &d, nullptr, nullptr, nullptr), GOC_SUCCESS);
          EXPECT_EQ(d, op & 1 ? exec_mask : 0);
          for (uint64_t mode : goc_test::dpp_modes) {
            d = 123;
            int status = floating[op](flags, exec_mask, mode, &d, nullptr, nullptr, nullptr);
            if (op >= 8) {
              EXPECT_EQ(status, GOC_ERROR_INVALID_FLAGS);
              EXPECT_EQ(d, 123U);
              continue;
            }
            ASSERT_EQ(status, GOC_SUCCESS);
            uint32_t expected = 0;
            for (unsigned lane = 0; lane < 32; ++lane) {
              int source;
              if ((op & 1) && goc_test::dpp_source(mode, exec_mask, lane, source))
                expected |= 1U << lane;
            }
            EXPECT_EQ(d, expected) << op << '/' << exec_mask << '/' << mode;
          }
        }
}

TEST(ConstantCompare, FloatingModifiersAndOptionalStateContract) {
  for (unsigned op = 0; op < 12; ++op) {
    uint64_t known = GOC_ALU_CLAMP | GOC_ALU_ABS_A | GOC_ALU_ABS_B | GOC_ALU_NEG_A | GOC_ALU_NEG_B;
    if (op < 4)
      known |= GOC_ALU_HIGH_A | GOC_ALU_HIGH_B;
    uint32_t d = 123, exceptions = 0x12340000;
    ASSERT_EQ(floating[op](GOC_FP_IEEE_MODE, 0x87654321, known, &d, nullptr, nullptr, &exceptions),
              GOC_SUCCESS);
    EXPECT_EQ(d, op & 1 ? 0x87654321U : 0U);
    EXPECT_EQ(exceptions, 0x12340000U);
    for (unsigned bit = 0; bit < 32; ++bit) {
      d = 123;
      uint64_t mode = 1ULL << bit;
      int status = floating[op](0, UINT32_MAX, mode, &d, nullptr, nullptr, nullptr);
      if (mode & known) {
        EXPECT_EQ(status, GOC_SUCCESS);
        EXPECT_EQ(d, op & 1 ? UINT32_MAX : 0);
      } else {
        EXPECT_EQ(status, GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(d, 123U);
      }
    }
    for (uint64_t strict : {uint64_t(0), GOC_SEMANTICS_STRICT}) {
      d = 123;
      EXPECT_EQ(floating[op](GOC_SEMANTICS_EXACT_EMPIRICAL | strict, UINT32_MAX, 0, &d, nullptr,
                             nullptr, &exceptions),
                GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
      EXPECT_EQ(d, 123U);
      EXPECT_EQ(exceptions, 0x12340000U);
    }
  }
}

TEST(ConstantCompareWave64, PredicatesMasksAndDppFiltering) {
  const decltype(&goc_v_cmp_f_i32_wave64) functions[] = {
      goc_v_cmp_f_i32_wave64,  goc_v_cmp_t_i32_wave64,  goc_v_cmpx_f_i32_wave64,
      goc_v_cmpx_t_i32_wave64, goc_v_cmp_f_u32_wave64,  goc_v_cmp_t_u32_wave64,
      goc_v_cmpx_f_u32_wave64, goc_v_cmpx_t_u32_wave64, goc_v_cmp_f_i64_wave64,
      goc_v_cmp_t_i64_wave64,  goc_v_cmpx_f_i64_wave64, goc_v_cmpx_t_i64_wave64,
      goc_v_cmp_f_u64_wave64,  goc_v_cmp_t_u64_wave64,  goc_v_cmpx_f_u64_wave64,
      goc_v_cmpx_t_u64_wave64};
  for (unsigned op = 0; op < 16; ++op)
    for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (uint64_t exec_mask :
             {uint64_t(0), UINT64_MAX, uint64_t(0xaaaaaaaaaaaaaaaaULL),
              uint64_t(0x8000000000000001ULL), uint64_t(0x00ff00ff00ff00ffULL)}) {
          uint64_t d = 123;
          ASSERT_EQ(functions[op](cpu | semantics, exec_mask, 0, &d, nullptr, nullptr),
                    GOC_SUCCESS);
          EXPECT_EQ(d, (op & 1) ? exec_mask : 0);
          for (auto mode : goc_test::dpp_modes) {
            d = 123;
            if (op >= 8) {
              EXPECT_EQ(functions[op](cpu | semantics, exec_mask, mode, &d, nullptr, nullptr),
                        GOC_ERROR_INVALID_FLAGS);
              EXPECT_EQ(d, 123U);
            } else {
              uint64_t expected = 0;
              for (unsigned lane = 0; lane < 64; ++lane) {
                int source;
                if ((op & 1) && goc_test::dpp_source(mode, exec_mask, lane, source))
                  expected |= 1ULL << lane;
              }
              ASSERT_EQ(functions[op](cpu | semantics, exec_mask, mode, &d, nullptr, nullptr),
                        GOC_SUCCESS);
              EXPECT_EQ(d, expected);
            }
          }
          for (unsigned bit = 0; bit < 32; ++bit) {
            d = 123;
            EXPECT_EQ(functions[op](cpu | semantics, exec_mask, 1ULL << bit, &d, nullptr, nullptr),
                      GOC_ERROR_INVALID_FLAGS);
            EXPECT_EQ(d, 123U);
          }
        }
}

namespace {

const decltype(&goc_v_cmp_f_f16_wave64) floating_wave64[] = {
    goc_v_cmp_f_f16_wave64,  goc_v_cmp_t_f16_wave64,  goc_v_cmpx_f_f16_wave64,
    goc_v_cmpx_t_f16_wave64, goc_v_cmp_f_f32_wave64,  goc_v_cmp_t_f32_wave64,
    goc_v_cmpx_f_f32_wave64, goc_v_cmpx_t_f32_wave64, goc_v_cmp_f_f64_wave64,
    goc_v_cmp_t_f64_wave64,  goc_v_cmpx_f_f64_wave64, goc_v_cmpx_t_f64_wave64};

} // namespace

TEST(ConstantCompareWave64, FloatingResultsMasksAndDpp) {
  for (unsigned op = 0; op < 12; ++op)
    for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (uint64_t exec_mask : {uint64_t(0), uint64_t(1), uint64_t(0x8000000000000000ULL),
                                   uint64_t(0xaaaaaaaaaaaaaaaaULL), UINT64_MAX}) {
          uint64_t flags = cpu | semantics | GOC_SEMANTICS_STRICT | GOC_FP_FLUSH_INPUT_DENORMALS |
                           GOC_FP_FLUSH_OUTPUT_DENORMALS | GOC_FP_IEEE_MODE;
          uint64_t d = 123;
          ASSERT_EQ(floating_wave64[op](flags, exec_mask, 0, &d, nullptr, nullptr, nullptr),
                    GOC_SUCCESS);
          EXPECT_EQ(d, op & 1 ? exec_mask : 0);
          for (uint64_t mode : goc_test::dpp_modes) {
            d = 123;
            int status = floating_wave64[op](flags, exec_mask, mode, &d, nullptr, nullptr, nullptr);
            if (op >= 8) {
              EXPECT_EQ(status, GOC_ERROR_INVALID_FLAGS);
              EXPECT_EQ(d, 123U);
              continue;
            }
            ASSERT_EQ(status, GOC_SUCCESS);
            uint64_t expected = 0;
            for (unsigned lane = 0; lane < 64; ++lane) {
              int source;
              if ((op & 1) && goc_test::dpp_source(mode, exec_mask, lane, source))
                expected |= 1ULL << lane;
            }
            EXPECT_EQ(d, expected) << op << '/' << exec_mask << '/' << mode;
          }
        }
}

TEST(ConstantCompareWave64, FloatingModifiersAndOptionalStateContract) {
  for (unsigned op = 0; op < 12; ++op) {
    uint64_t known = GOC_ALU_CLAMP | GOC_ALU_ABS_A | GOC_ALU_ABS_B | GOC_ALU_NEG_A | GOC_ALU_NEG_B;
    if (op < 4)
      known |= GOC_ALU_HIGH_A | GOC_ALU_HIGH_B;
    uint64_t d = 123;
    uint32_t exceptions = 0x12340000;
    ASSERT_EQ(floating_wave64[op](GOC_FP_IEEE_MODE, 0x8765432187654321ULL, known, &d, nullptr,
                                  nullptr, &exceptions),
              GOC_SUCCESS);
    EXPECT_EQ(d, op & 1 ? 0x8765432187654321ULL : 0U);
    EXPECT_EQ(exceptions, 0x12340000U);
    for (unsigned bit = 0; bit < 32; ++bit) {
      d = 123;
      uint64_t mode = 1ULL << bit;
      int status = floating_wave64[op](0, UINT64_MAX, mode, &d, nullptr, nullptr, nullptr);
      if (mode & known) {
        EXPECT_EQ(status, GOC_SUCCESS);
        EXPECT_EQ(d, op & 1 ? UINT64_MAX : 0);
      } else {
        EXPECT_EQ(status, GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(d, 123U);
      }
    }
    for (uint64_t strict : {uint64_t(0), GOC_SEMANTICS_STRICT}) {
      d = 123;
      EXPECT_EQ(floating_wave64[op](GOC_SEMANTICS_EXACT_EMPIRICAL | strict, UINT64_MAX, 0, &d,
                                    nullptr, nullptr, &exceptions),
                GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
      EXPECT_EQ(d, 123U);
      EXPECT_EQ(exceptions, 0x12340000U);
    }
  }
}

TEST(ConstantCompare, ExactStateRejectionPrecedesEmptyExecAndFlagValidation) {
  for (unsigned op = 0; op < 12; ++op)
    for (uint64_t strict : {uint64_t(0), GOC_SEMANTICS_STRICT})
      for (uint64_t bad_flags : {0ULL, 1ULL << 63}) {
        uint64_t flags = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_FP_IEEE_MODE | strict | bad_flags;
        uint32_t d32 = 0x12345678, trapsts = 0x87654321;
        uint64_t d64 = 0x0123456789abcdefULL;
        EXPECT_EQ(floating[op](flags, 0, bad_flags, &d32, nullptr, nullptr, &trapsts),
                  GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
        EXPECT_EQ(floating_wave64[op](flags, 0, bad_flags, &d64, nullptr, nullptr, &trapsts),
                  GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
        EXPECT_EQ(d32, 0x12345678U);
        EXPECT_EQ(d64, 0x0123456789abcdefULL);
        EXPECT_EQ(trapsts, 0x87654321U);
      }
}

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

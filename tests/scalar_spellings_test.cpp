// SPDX-License-Identifier: MIT

#include "goc/goc.h"

#include <gtest/gtest.h>
#include <stdint.h>

TEST(ScalarSpellings, ArithmeticResultsAndScc) {
  const uint32_t values[] = {0,          1,          2,          0x7ffffffe, 0x7fffffff,
                             0x80000000, 0x80000001, 0xfffffffe, 0xffffffff};
  for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
    for (uint32_t a : values)
      for (uint32_t b : values)
        for (unsigned op = 0; op < 6; ++op)
          for (uint32_t carry : {0U, 1U, 0xaaaa0000U, 0xffffffffU}) {
            uint32_t d = 0, scc = 9;
            uint64_t flags = semantics | GOC_SEMANTICS_STRICT;
            auto call = [&](uint32_t *dst, uint32_t *cc) {
              switch (op) {
              case 0:
                return goc_s_add_u32(flags, 0, dst, a, b, cc);
              case 1:
                return goc_s_sub_u32(flags, 0, dst, a, b, cc);
              case 2:
                return goc_s_add_i32(flags, 0, dst, a, b, cc);
              case 3:
                return goc_s_sub_i32(flags, 0, dst, a, b, cc);
              case 4:
                return goc_s_addc_u32(flags, 0, dst, a, b, carry, cc);
              default:
                return goc_s_subb_u32(flags, 0, dst, a, b, carry, cc);
              }
            };
            int64_t x = a, y = b;
            if (op == 2 || op == 3) {
              if (a >> 31)
                x -= 1LL << 32;
              if (b >> 31)
                y -= 1LL << 32;
            }
            if (op >= 4)
              y += carry & 1;
            int64_t result = op % 2 ? x - y : x + y;
            bool condition = (op == 2 || op == 3) ? result < -2147483648LL || result > 2147483647LL
                             : op % 2             ? result < 0
                                                  : result > 4294967295LL;
            ASSERT_EQ(call(&d, &scc), GOC_SUCCESS);
            ASSERT_EQ(d, uint32_t(result));
            ASSERT_EQ(scc, uint32_t(condition));
            ASSERT_EQ(call(&d, &d), GOC_SUCCESS);
            ASSERT_EQ(d, uint32_t(condition));
          }
}

TEST(ScalarSpellings, EveryImmediateComparisonAndAdd) {
  using Compare = int (*)(uint64_t, uint64_t, uint32_t *, uint32_t, uint16_t);
  const Compare compare[2][6] = {{goc_s_cmpk_eq_i32, goc_s_cmpk_lg_i32, goc_s_cmpk_gt_i32,
                                  goc_s_cmpk_ge_i32, goc_s_cmpk_lt_i32, goc_s_cmpk_le_i32},
                                 {goc_s_cmpk_eq_u32, goc_s_cmpk_lg_u32, goc_s_cmpk_gt_u32,
                                  goc_s_cmpk_ge_u32, goc_s_cmpk_lt_u32, goc_s_cmpk_le_u32}};
  for (uint32_t immediate = 0; immediate < 65536; ++immediate) {
    int64_t signed_imm = immediate < 32768 ? int64_t(immediate) : int64_t(immediate) - 65536;
    for (uint32_t a :
         {0U, 0x7fffffffU, 0x80000000U, 0xffffffffU, immediate, uint32_t(signed_imm)}) {
      int64_t signed_a = a < 0x80000000U ? int64_t(a) : int64_t(a) - (1LL << 32);
      uint32_t d = a, cc = 9;
      int64_t sum = signed_a + signed_imm;
      ASSERT_EQ(goc_s_addk_i32(GOC_SEMANTICS_EXACT_EMPIRICAL, 0, &d, immediate, &cc), GOC_SUCCESS);
      ASSERT_EQ(d, uint32_t(sum));
      ASSERT_EQ(cc, uint32_t(sum < -2147483648LL || sum > 2147483647LL));
      for (unsigned u = 0; u < 2; ++u) {
        int64_t left = u ? int64_t(a) : signed_a, right = u ? int64_t(immediate) : signed_imm;
        bool wanted[] = {left == right, left != right, left > right,
                         left >= right, left < right,  left <= right};
        for (unsigned op = 0; op < 6; ++op) {
          ASSERT_EQ(compare[u][op](GOC_SEMANTICS_EXACT_EMPIRICAL, 0, &cc, a, immediate),
                    GOC_SUCCESS);
          ASSERT_EQ(cc, uint32_t(wanted[op]));
        }
      }
    }
  }
}

TEST(ScalarSpellings, ErrorsPreserveOutputs) {
  for (uint64_t mode : {1ULL, 1ULL << 31, 1ULL << 63}) {
    uint32_t d = 123, cc = 456;
    EXPECT_EQ(goc_s_addk_i32(0, mode, &d, 0xffff, &cc), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_s_addc_u32(0, mode, &d, 0xffffffff, 1, 1, &cc), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_s_subb_u32(0, mode, &d, 0, 1, 1, &cc), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_s_cmpk_eq_i32(0, mode, &cc, 0, 0), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_s_cmpk_eq_u32(0, mode, &cc, 0, 0), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(d, 123U);
    EXPECT_EQ(cc, 456U);
  }
}

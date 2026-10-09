// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "internal.h"
#include "rdna4_dpp_reference.h"
#include "rdna4_exec_masks.h"
#include "rdna4_unary_reference.h"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <stdint.h>

TEST(DppUnary, HardwareCorpus) {
  // Gfx1201, MODE=0xf0: 11 instructions x 7 descriptors x 2 modifier
  // settings x 8 EXEC masks x 32 lanes. NaN payloads are normalized.
  const float inputs[] = {0, 1, 4, 16, 64, 256, -1, -4};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (uint32_t mask :
         {0xffffffffu, 0u, 0xaaaaaaaau, 0x55555555u, 1u, 0x80000000u, 0xffffu, 0xffff0000u})
      for (auto fn : goc_test::unary_functions)
        for (uint64_t descriptor : goc_test::dpp_modes)
          for (unsigned modified = 0; modified < 2; ++modified) {
            uint32_t a[32], d[32];
            for (unsigned lane = 0; lane < 32; ++lane) {
              a[lane] = goc::as_bits(inputs[lane % 8]);
              d[lane] = goc::as_bits(float(100 + lane));
            }
            auto pa = a, pd = d;
            uint64_t mode = descriptor | (modified ? GOC_ALU_NEG_A | GOC_ALU_OMOD_2 : 0);
            ASSERT_EQ(fn(cpu, mask, mode, &pd, &pa), GOC_SUCCESS);
            for (uint32_t value : d) {
              if ((value & 0x7fffffff) > 0x7f800000)
                value = 0x7fc00000;
              hash = (hash ^ value) * UINT64_C(1099511628211);
            }
          }
    EXPECT_EQ(hash, UINT64_C(0x59f5427020212b25));
  }
}

TEST(DppUnary, AllModifiersMasksAliasesAndSpecialValues) {
  const uint32_t inputs[] = {0,          0x80000000, 1,          0x80000001, 0x007fffff,
                             0x807fffff, 0x00800000, 0x80800000, 0x00ffffff, 0x80ffffff,
                             0x3f000000, 0xbf000000, 0x3fc00000, 0xbfc00000, 0x3f800000,
                             0xbf800000, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
                             0x7fc12345, 0xffc12345, 0x7f812345, 0xff812345};
  auto masks = rdna4_exec_masks();
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned op = 0; op < 11; ++op)
      for (unsigned variant = 0; variant < 32; ++variant)
        for (uint64_t descriptor : goc_test::dpp_modes)
          for (uint32_t mask : masks)
            for (bool alias : {false, true}) {
              SCOPED_TRACE(::testing::Message() << cpu << '/' << op << '/' << variant << '/'
                                                << descriptor << '/' << mask << '/' << alias);
              uint32_t a[34], d[34], before[32];
              std::fill(a, a + 34, 0xdeadbeef);
              std::fill(d, d + 34, 0xdeadbeef);
              for (unsigned lane = 0; lane < 32; ++lane)
                a[lane + 1] = before[lane] = inputs[(lane + variant) % 24];
              uint32_t *pa = a + 1, *pd = alias ? a + 1 : d + 1;
              uint32_t low = (variant & 1 ? GOC_ALU_NEG_A : 0) | (variant & 2 ? GOC_ALU_ABS_A : 0) |
                             ((variant >> 2 & 3) << 6) | (variant & 16 ? GOC_ALU_CLAMP : 0);
              ASSERT_EQ(goc_test::unary_functions[op](cpu, mask, descriptor | low, &pd, &pa),
                        GOC_SUCCESS);
              for (unsigned lane = 0; lane < 32; ++lane) {
                int source;
                if (!goc_test::dpp_source(descriptor, mask, lane, source)) {
                  ASSERT_EQ(pd[lane], alias ? before[lane] : 0xdeadbeefu);
                  continue;
                }
                float want = goc_test::unary_reference(
                    op, source < 0 ? 0 : goc::as_float(before[source]), low);
                float got = goc::as_float(pd[lane]);
                if (std::isnan(want)) {
                  ASSERT_TRUE(std::isnan(got));
                } else if (want == 0 || std::isinf(want)) {
                  ASSERT_EQ(pd[lane], goc::as_bits(want));
                } else {
                  ASSERT_FLOAT_EQ(got, want);
                }
              }
              ASSERT_EQ(a[0], 0xdeadbeefu);
              ASSERT_EQ(a[33], 0xdeadbeefu);
              ASSERT_EQ(d[0], 0xdeadbeefu);
              ASSERT_EQ(d[33], 0xdeadbeefu);
            }
}

TEST(DppUnary, ValidationBeforeOperandAccess) {
  const uint64_t bad[] = {GOC_DPP8 | GOC_DPP16,
                          GOC_DPP_FI,
                          GOC_DPP8 | GOC_DPP_BOUND_CTRL,
                          GOC_DPP16 | (UINT64_C(0x110) << GOC_DPP_CTRL_SHIFT),
                          GOC_DPP16 | (UINT64_C(1) << 63),
                          GOC_DPP8 | GOC_ALU_NEG_B};
  for (auto fn : goc_test::unary_functions)
    for (uint32_t mask : {UINT32_C(0), UINT32_MAX}) {
      for (uint64_t mode : bad) {
        EXPECT_EQ(fn(0, mask, mode, nullptr, nullptr), GOC_ERROR_INVALID_FLAGS);
      }
      for (uint64_t mode : goc_test::dpp_modes) {
        EXPECT_EQ(fn(UINT64_C(1) << 63, mask, mode, nullptr, nullptr), GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(
            fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, mask, mode, nullptr, nullptr),
            GOC_ERROR_UNSUPPORTED_SEMANTICS);
        EXPECT_EQ(fn(0, 0, mode, nullptr, nullptr), GOC_SUCCESS);
      }
    }
}

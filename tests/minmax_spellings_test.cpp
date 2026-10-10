// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "minmax_reference.h"

#include <algorithm>
#include <gtest/gtest.h>
#include <stdint.h>

namespace {

using Binary = decltype(&goc_v_min_f32);
using Ternary = decltype(&goc_v_min3_f32);
const Binary binary[] = {goc_v_min_f16, goc_v_max_f16, goc_v_min_f32,    goc_v_max_f32,
                         goc_v_min_f64, goc_v_max_f64, goc_v_pk_min_f16, goc_v_pk_max_f16};
const Ternary ternary[] = {goc_v_min3_f16,   goc_v_max3_f16, goc_v_minmax_f16, goc_v_maxmin_f16,
                           goc_v_med3_f16,   goc_v_min3_f32, goc_v_max3_f32,   goc_v_minmax_f32,
                           goc_v_maxmin_f32, goc_v_med3_f32};

} // namespace

TEST(MinmaxSpellings, BinaryFiniteResultsMasksAndAliases) {
  // -2 versus +3, with +5 versus -4 in the upper packed half.
  const uint32_t a_words[] = {0x4500c000, 0xc0000000, 0, 0x4500c000};
  const uint32_t b_words[] = {0xc4004200, 0x40400000, 0, 0xc4004200};
  for (unsigned op = 0; op < 8; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint32_t exec_mask : {0U, 0xa55aa55aU, UINT32_MAX})
        for (bool alias : {false, true}) {
          SCOPED_TRACE(::testing::Message()
                       << op << '/' << cpu << '/' << exec_mask << '/' << alias);
          unsigned type = op / 2;
          bool maximum = op & 1;
          uint32_t a[2][32], b[2][32], out[2][32];
          for (unsigned lane = 0; lane < 32; ++lane) {
            a[0][lane] = a_words[type];
            b[0][lane] = b_words[type];
            a[1][lane] = 0xc0000000;
            b[1][lane] = 0x40080000;
            out[0][lane] = out[1][lane] = 0xdeadbeef;
          }
          uint32_t *d[] = {alias ? a[0] : out[0], alias ? a[1] : out[1]};
          const uint32_t *pa[] = {a[0], a[1]}, *pb[] = {b[0], b[1]};
          uint32_t state = 0x12345678;
          ASSERT_EQ(binary[op](cpu, exec_mask, 0, d, pa, pb, &state), GOC_SUCCESS);
          EXPECT_EQ(state, 0x12345678U);
          for (unsigned lane = 0; lane < 32; ++lane) {
            uint32_t before = alias ? a_words[type] : 0xdeadbeef;
            uint32_t result = maximum ? b_words[type] : a_words[type];
            if (type == 0)
              result = (before & 0xffff0000) | (result & 0xffff);
            if (type == 3)
              result = maximum ? 0x45004200 : 0xc400c000;
            EXPECT_EQ(d[0][lane], (exec_mask >> lane & 1) ? result : before);
            if (type == 2) {
              EXPECT_EQ(d[1][lane], (exec_mask >> lane & 1) ? (maximum ? 0x40080000U : 0xc0000000U)
                                                            : (alias ? 0xc0000000U : 0xdeadbeefU));
            }
          }
        }
}

TEST(MinmaxSpellings, TernaryFiniteOrderAndModifiers) {
  const uint32_t values[] = {0xc0000000, 0x40400000, 0x3f800000};
  const uint32_t halves[] = {0xc000, 0x4200, 0x3c00};
  const int reference_ops[] = {0, 1, 2, 3, 8};
  for (unsigned op = 0; op < 10; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint32_t mode : {0U, uint32_t(GOC_ALU_NEG_A), uint32_t(GOC_ALU_ABS_B)}) {
        uint32_t a[32], b[32], c[32], d[32];
        for (unsigned lane = 0; lane < 32; ++lane) {
          a[lane] = op < 5 ? halves[lane % 3] : values[lane % 3];
          b[lane] = op < 5 ? halves[(lane + 1) % 3] : values[(lane + 1) % 3];
          c[lane] = op < 5 ? halves[(lane + 2) % 3] : values[(lane + 2) % 3];
          d[lane] = 0;
        }
        auto pa = a, pb = b, pc = c, pd = d;
        ASSERT_EQ(ternary[op](cpu, UINT32_MAX, mode, &pd, &pa, &pb, &pc, nullptr), GOC_SUCCESS);
        for (unsigned lane = 0; lane < 32; ++lane) {
          uint32_t expected = goc_test::minmax_reference::reference(
              reference_ops[op % 5], values[lane % 3], values[(lane + 1) % 3],
              values[(lane + 2) % 3], mode);
          // All selected values are normal powers/small integers, exactly representable in half.
          if (op < 5)
            expected = ((expected >> 16) & 0x8000) | (((expected >> 23 & 255) - 112) << 10) |
                       ((expected >> 13) & 1023);
          EXPECT_EQ(d[lane], expected);
        }
      }
}

TEST(MinmaxSpellings, ExactReportingErrorsPreserveOutputs) {
  uint32_t a[32] = {}, d[32];
  std::fill(d, d + 32, 0xdeadbeef);
  auto pa = a, pd = d;
  uint32_t state = 0x12345678;
  for (auto fn : binary) {
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, UINT32_MAX, 0, &pd, &pa, &pa,
                 nullptr),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, 0, &pd, &pa, &pa, &state),
              GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
  }
  for (auto fn : ternary) {
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, UINT32_MAX, 0, &pd, &pa, &pa,
                 &pa, nullptr),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, 0, &pd, &pa, &pa, &pa, &state),
              GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
  }
  EXPECT_EQ(state, 0x12345678U);
  for (auto value : d)
    EXPECT_EQ(value, 0xdeadbeefU);
}

TEST(MinmaxSpellings, BinaryNumberSelectionSpecialValues) {
  // A, B, minimum, maximum: default number selection, not IEEE-mode propagation.
  const uint32_t cases[][4] = {
      {0, 0x80000000, 0x80000000, 0},
      {0x7fc12345, 0x3f800000, 0x3f800000, 0x3f800000},
      {0x3f800000, 0x7f812345, 0x3f800000, 0x3f800000},
      {0xff800000, 0x7f800000, 0xff800000, 0x7f800000},
      {1, 0, 0, 1},
  };
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned maximum = 0; maximum < 2; ++maximum) {
      uint32_t a[32], b[32], d[32];
      for (unsigned lane = 0; lane < 32; ++lane) {
        a[lane] = cases[lane % 5][0];
        b[lane] = cases[lane % 5][1];
      }
      auto pa = a, pb = b, pd = d;
      ASSERT_EQ(binary[2 + maximum](cpu, UINT32_MAX, 0, &pd, &pa, &pb, nullptr), GOC_SUCCESS);
      for (unsigned lane = 0; lane < 32; ++lane)
        EXPECT_EQ(d[lane], cases[lane % 5][2 + maximum]);
    }
}

// SPDX-License-Identifier: MIT

#include "gfx11_dot2_fixtures.h"
#include "goc/goc.h"

#include <algorithm>
#include <gtest/gtest.h>
#include <stdint.h>

namespace {

int execute(unsigned op, uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b, uint32_t *excp_flag_user) {
  const uint32_t literal = 0x40800000; // 4.0f
  const uint32_t condition = 0xaaaaaaaa;
  switch (op) {
  case 0:
    return goc_v_dual_fmac_f32(flags, exec_mask, mode, d, a, b, excp_flag_user);
  case 1:
    return goc_v_dual_fmaak_f32(flags, exec_mask, mode, d, a, b, literal, excp_flag_user);
  case 2:
    return goc_v_dual_fmamk_f32(flags, exec_mask, mode, d, a, literal, b, excp_flag_user);
  case 3:
    return goc_v_dual_mul_f32(flags, exec_mask, mode, d, a, b, excp_flag_user);
  case 4:
    return goc_v_dual_add_f32(flags, exec_mask, mode, d, a, b, excp_flag_user);
  case 5:
    return goc_v_dual_sub_f32(flags, exec_mask, mode, d, a, b, excp_flag_user);
  case 6:
    return goc_v_dual_subrev_f32(flags, exec_mask, mode, d, a, b, excp_flag_user);
  case 7:
    return goc_v_dual_mul_dx9_zero_f32(flags, exec_mask, mode, d, a, b, excp_flag_user);
  case 8:
    return goc_v_dual_mov_b32(flags, exec_mask, mode, d, a);
  case 9:
    return goc_v_dual_cndmask_b32(flags, exec_mask, mode, d, a, b, condition);
  case 10:
    return goc_v_dual_max_num_f32(flags, exec_mask, mode, d, a, b, excp_flag_user);
  case 11:
    return goc_v_dual_min_num_f32(flags, exec_mask, mode, d, a, b, excp_flag_user);
  case 12:
    return goc_v_dual_max_f32(flags, exec_mask, mode, d, a, b, excp_flag_user);
  case 13:
    return goc_v_dual_min_f32(flags, exec_mask, mode, d, a, b, excp_flag_user);
  case 14:
    return goc_v_dual_add_nc_u32(flags, exec_mask, mode, d, a, b);
  case 15:
    return goc_v_dual_lshlrev_b32(flags, exec_mask, mode, d, a, b);
  case 16:
    return goc_v_dual_and_b32(flags, exec_mask, mode, d, a, b);
  case 17:
    return goc_v_dual_dot2acc_f32_f16(flags, exec_mask, mode, d, a, b, excp_flag_user);
  case 18:
    return goc_v_dual_dot2acc_f32_bf16(flags, exec_mask, mode, d, a, b, excp_flag_user);
  case 19:
    return goc_v_dual_dot2acc_f32_f16_rdna4(flags, exec_mask, mode, d, a, b, excp_flag_user);
  case 20:
    return goc_v_dual_dot2acc_f32_bf16_rdna4(flags, exec_mask, mode, d, a, b, excp_flag_user);
  default:
    return -1;
  }
}

} // namespace

TEST(Dual, IndependentFiniteResultsAndMasks) {
  // Default A=2, B=3, old D=4. Literal FMA uses 4.0f.
  const uint32_t expected[] = {
      0x41200000, 0x41200000, 0x41300000, 0x40c00000, 0x40a00000, 0xbf800000, 0x3f800000,
      0x40c00000, 0x40000000, 0,          0x40400000, 0x40000000, 0x40400000, 0x40000000,
      5,          12,         2,          0x40a00000, 0x40a00000, 0x40a00000, 0x40a00000};
  for (unsigned op = 0; op < 21; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint32_t exec_mask : {0U, 1U, 0x80000000U, 0xaaaaaaaaU, UINT32_MAX}) {
        uint32_t a[32], b[32], d[32], before;
        uint32_t av = 0x40000000, bv = 0x40400000;
        before = 0x40800000;
        if (op >= 14 && op <= 16) {
          av = 2;
          bv = 3;
        }
        if (op >= 17) {
          av = (op & 1) ? 0x3c003c00 : 0x3f803f80;
          bv = 0x40004000;
          before = 0x3f800000;
        }
        std::fill(a, a + 32, av);
        std::fill(b, b + 32, bv);
        std::fill(d, d + 32, before);
        auto pa = a, pb = b, pd = d;
        uint32_t state = 0x80000000;
        ASSERT_EQ(execute(op, cpu, exec_mask, 0, &pd, &pa, &pb, &state), GOC_SUCCESS);
        for (unsigned lane = 0; lane < 32; ++lane) {
          uint32_t want = op == 9 ? (lane & 1 ? bv : av) : expected[op];
          EXPECT_EQ(d[lane], (exec_mask >> lane & 1) ? want : before) << op << '/' << lane;
        }
        EXPECT_EQ(state, 0x80000000U);
      }
}

TEST(Dual, NoEncodedModifiersAndErrorsPreserveState) {
  for (unsigned op = 0; op < 21; ++op)
    for (unsigned bit = 0; bit < 64; ++bit) {
      uint32_t a[32] = {}, d[32];
      std::fill(d, d + 32, 123);
      auto pa = a, pd = d;
      uint32_t state = 0x80000000;
      EXPECT_EQ(execute(op, 0, UINT32_MAX, 1ULL << bit, &pd, &pa, &pa, &state),
                GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(state, 0x80000000U);
      for (auto word : d)
        EXPECT_EQ(word, 123U);
    }
}

TEST(Dual, LegalPairExecutesXBeforeY) {
  // X: add v0,v3,v4; Y: mov v3,v5. X reads old v3 before Y overwrites it.
  // X/Y destinations have opposite parity, and source-0 banks differ (3 vs 1).
  uint32_t x[32], y[32], b[32], source_y[32];
  std::fill(x, x + 32, 0);
  std::fill(y, y + 32, 0x40000000);
  std::fill(b, b + 32, 0x40400000);
  std::fill(source_y, source_y + 32, 0x40e00000);
  auto px = x, py = y, pb = b, ps = source_y;
  ASSERT_EQ(goc_v_dual_add_f32(0, UINT32_MAX, 0, &px, &py, &pb, nullptr), GOC_SUCCESS);
  ASSERT_EQ(goc_v_dual_mov_b32(0, UINT32_MAX, 0, &py, &ps), GOC_SUCCESS);
  for (unsigned lane = 0; lane < 32; ++lane) {
    EXPECT_EQ(x[lane], 0x40a00000U);
    EXPECT_EQ(y[lane], 0x40e00000U);
  }
}

TEST(Dual, Rdna3ExactDotCapturesAndArchitectureDifference) {
  const uint64_t exact = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;
  for (bool bf16 : {false, true}) {
    auto fn = bf16 ? goc_v_dual_dot2acc_f32_bf16 : goc_v_dual_dot2acc_f32_f16;
    const auto *cases = bf16 ? kDot2BF16Cases.data() : kDot2F16Cases.data();
    size_t count = bf16 ? kDot2BF16Cases.size() : kDot2F16Cases.size();
    for (size_t base = 0; base < count; base += 32) {
      uint32_t a[32], b[32], d[32];
      for (unsigned lane = 0; lane < 32; ++lane) {
        const auto &c = cases[(base + lane) % count];
        a[lane] = c.a;
        b[lane] = c.b;
        d[lane] = c.c;
      }
      auto pa = a, pb = b, pd = d;
      ASSERT_EQ(fn(exact, UINT32_MAX, 0, &pd, &pa, &pb, nullptr), GOC_SUCCESS);
      for (unsigned lane = 0; lane < 32; ++lane)
        EXPECT_EQ(d[lane], cases[(base + lane) % count].expected);
    }
  }
  uint32_t a[32], b[32], d3[32], d4[32];
  std::fill(a, a + 32, kDot2F16Cases[0].a);
  std::fill(b, b + 32, kDot2F16Cases[0].b);
  std::fill(d3, d3 + 32, kDot2F16Cases[0].c);
  std::fill(d4, d4 + 32, kDot2F16Cases[0].c);
  auto pa = a, pb = b, p3 = d3, p4 = d4;
  ASSERT_EQ(goc_v_dual_dot2acc_f32_f16(exact, UINT32_MAX, 0, &p3, &pa, &pb, nullptr), GOC_SUCCESS);
  uint32_t exceptions = 0x80000000;
  ASSERT_EQ(goc_v_dual_dot2acc_f32_f16_rdna4(exact, UINT32_MAX, 0, &p4, &pa, &pb, &exceptions),
            GOC_SUCCESS);
  EXPECT_EQ(d3[0], 0x425c0001U);
  EXPECT_NE(d3[0], d4[0]);
}

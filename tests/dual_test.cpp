// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "exec_masks.h"
#include "gfx11_dot2_fixtures.h"
#include "goc/goc.h"

#include <algorithm>
#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>

namespace {

int execute(unsigned op, uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b, uint32_t *state, bool rdna3 = false,
            uint32_t literal = 0x40800000) {
  const uint32_t condition = 0xaaaaaaaa;
  switch (op) {
  case 0:
    return (rdna3 ? goc_v_dual_fmac_f32 : goc_v_dual_fmac_f32_rdna4)(flags, exec_mask, mode, d, a,
                                                                     b, state);
  case 1:
    return (rdna3 ? goc_v_dual_fmaak_f32 : goc_v_dual_fmaak_f32_rdna4)(flags, exec_mask, mode, d, a,
                                                                       b, literal, state);
  case 2:
    return (rdna3 ? goc_v_dual_fmamk_f32 : goc_v_dual_fmamk_f32_rdna4)(flags, exec_mask, mode, d, a,
                                                                       literal, b, state);
  case 3:
    return (rdna3 ? goc_v_dual_mul_f32 : goc_v_dual_mul_f32_rdna4)(flags, exec_mask, mode, d, a, b,
                                                                   state);
  case 4:
    return (rdna3 ? goc_v_dual_add_f32 : goc_v_dual_add_f32_rdna4)(flags, exec_mask, mode, d, a, b,
                                                                   state);
  case 5:
    return (rdna3 ? goc_v_dual_sub_f32 : goc_v_dual_sub_f32_rdna4)(flags, exec_mask, mode, d, a, b,
                                                                   state);
  case 6:
    return (rdna3 ? goc_v_dual_subrev_f32 : goc_v_dual_subrev_f32_rdna4)(flags, exec_mask, mode, d,
                                                                         a, b, state);
  case 7:
    return (rdna3 ? goc_v_dual_mul_dx9_zero_f32
                  : goc_v_dual_mul_dx9_zero_f32_rdna4)(flags, exec_mask, mode, d, a, b, state);
  case 8:
    return goc_v_dual_mov_b32(flags, exec_mask, mode, d, a);
  case 9:
    return goc_v_dual_cndmask_b32(flags, exec_mask, mode, d, a, b, condition);
  case 10:
    return goc_v_dual_max_num_f32(flags, exec_mask, mode, d, a, b, state);
  case 11:
    return goc_v_dual_min_num_f32(flags, exec_mask, mode, d, a, b, state);
  case 12:
    return goc_v_dual_max_f32(flags, exec_mask, mode, d, a, b, state);
  case 13:
    return goc_v_dual_min_f32(flags, exec_mask, mode, d, a, b, state);
  case 14:
    return goc_v_dual_add_nc_u32(flags, exec_mask, mode, d, a, b);
  case 15:
    return goc_v_dual_lshlrev_b32(flags, exec_mask, mode, d, a, b);
  case 16:
    return goc_v_dual_and_b32(flags, exec_mask, mode, d, a, b);
  case 17:
    return goc_v_dual_dot2acc_f32_f16(flags, exec_mask, mode, d, a, b, state);
  case 18:
    return goc_v_dual_dot2acc_f32_bf16(flags, exec_mask, mode, d, a, b, state);
  case 19:
    return goc_v_dual_dot2acc_f32_f16_rdna4(flags, exec_mask, mode, d, a, b, state);
  case 20:
    return goc_v_dual_dot2acc_f32_bf16_rdna4(flags, exec_mask, mode, d, a, b, state);
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
  ASSERT_EQ(goc_v_dual_add_f32_rdna4(0, UINT32_MAX, 0, &px, &py, &pb, nullptr), GOC_SUCCESS);
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

TEST(Dual, FusedArchitectureNaNPriorityAndReporting) {
  for (unsigned op = 0; op < 3; ++op)
    for (bool rdna3 : {false, true})
      for (bool ieee : {false, true})
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
          for (int alias = 0; alias < 3; ++alias) {
            uint32_t words[3][32], before[3][32];
            std::fill(words[0], words[0] + 32, 0xff800123);
            std::fill(words[1], words[1] + 32, 0x3f800000);
            std::fill(words[2], words[2] + 32, 0x3f800000);
            std::memcpy(before, words, sizeof(words));
            uint32_t *d = words[alias];
            const uint32_t *a = words[0], *b = words[1];
            uint64_t flags = cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT |
                             (ieee ? GOC_FP_IEEE_MODE : 0);
            uint32_t state = 0x40000000;
            if (rdna3) {
              EXPECT_EQ(execute(op, flags, UINT32_MAX, 0, &d, &a, &b, &state, true),
                        GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
              EXPECT_EQ(std::memcmp(before, words, sizeof(words)), 0);
              EXPECT_EQ(state, 0x40000000U);
            }
            ASSERT_EQ(execute(op, flags, 0xa5a55a5a, 0, &d, &a, &b, nullptr, rdna3), GOC_SUCCESS);
            EXPECT_EQ(state, 0x40000000U);
            for (unsigned lane = 0; lane < 32; ++lane)
              EXPECT_EQ(d[lane], (0xa5a55a5aU >> lane) & 1
                                     ? rdna3 && !ieee ? 0xff800123U : 0xffc00123U
                                     : before[alias][lane]);
          }
}

TEST(Dual, FloatingGuestPoliciesMatchTheirArchitecture) {
  const uint32_t values[] = {0,          0x80000000, 1,          0x80000001, 0x7fffff,   0x800000,
                             0x3f800000, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7f800001,
                             0xffc00123, 0x3f800001, 0x3f7fffff, 0xbf800000};
  for (unsigned op = 0; op < 8; ++op)
    for (bool rdna3 : {false, true})
      for (unsigned policy = 0; policy < 8; ++policy)
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
          for (int alias = 0; alias < 3; ++alias) {
            uint64_t flags = cpu | (policy & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                             (policy & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) |
                             (policy & 4 ? GOC_FP_IEEE_MODE : 0);
            uint32_t words[3][32], expected[32];
            for (unsigned reg = 0; reg < 3; ++reg)
              for (unsigned lane = 0; lane < 32; ++lane)
                words[reg][lane] = values[(lane + reg * 7) % 16];
            std::memcpy(expected, words[alias], sizeof(expected));
            uint32_t *d = words[alias], *e = expected;
            const uint32_t *a = words[0], *b = words[1];
            uint32_t exec_mask = 0xa7109efa, state = 0x12345678;
            int reference;
            if (op == 0)
              reference = (rdna3 ? goc_v_fmac_f32 : goc_v_fmac_f32_rdna4)(flags, exec_mask, 0, &e,
                                                                          &a, &b, nullptr);
            else if (op == 1)
              reference = (rdna3 ? goc_v_fmaak_f32 : goc_v_fmaak_f32_rdna4)(
                  flags, exec_mask, 0, &e, &a, &b, 0x40800000, nullptr);
            else if (op == 2)
              reference = (rdna3 ? goc_v_fmamk_f32 : goc_v_fmamk_f32_rdna4)(
                  flags, exec_mask, 0, &e, &a, 0x40800000, &b, nullptr);
            else {
              using Fn = decltype(&goc_v_mul_f32);
              const Fn functions[2][5] = {{goc_v_mul_f32_rdna4, goc_v_add_f32_rdna4,
                                           goc_v_sub_f32_rdna4, goc_v_subrev_f32_rdna4,
                                           goc_v_mul_dx9_zero_f32_rdna4},
                                          {goc_v_mul_f32, goc_v_add_f32, goc_v_sub_f32,
                                           goc_v_subrev_f32, goc_v_mul_dx9_zero_f32}};
              reference = functions[rdna3][op - 3](flags, exec_mask, 0, &e, &a, &b, nullptr);
            }
            ASSERT_EQ(reference, GOC_SUCCESS);
            ASSERT_EQ(execute(op, flags, exec_mask, 0, &d, &a, &b, &state, rdna3), GOC_SUCCESS);
            EXPECT_EQ(state, 0x12345678U);
            for (unsigned lane = 0; lane < 32; ++lane)
              EXPECT_EQ(d[lane], expected[lane]);
            std::memcpy(expected, d, sizeof(expected));
            EXPECT_EQ(execute(op, flags, exec_mask, GOC_ALU_NEG_A, &d, &a, &b, &state, rdna3),
                      GOC_ERROR_INVALID_FLAGS);
            EXPECT_EQ(state, 0x12345678U);
            EXPECT_EQ(std::memcmp(expected, d, sizeof(expected)), 0);
          }
}

TEST(Dual, IntegerBitPatternsShiftCountsAndAliases) {
  for (unsigned op : {8U, 9U, 14U, 15U, 16U})
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
        for (unsigned seed = 0; seed < 32; ++seed)
          for (unsigned breg : {0U, 1U})
            for (unsigned target = 0; target < 3; ++target)
              for (uint32_t exec_mask : exec_masks()) {
                uint32_t words[3][34], expected[3][34];
                for (unsigned reg = 0; reg < 3; ++reg)
                  for (unsigned lane = 0; lane < 34; ++lane) {
                    // Each batch covers every shift count; high count bits vary.
                    uint32_t value = reg == 0 ? 0xffffffe0U | ((lane + seed) & 31)
                                              : (1U << seed) ^ (lane & 1 ? UINT32_MAX : 0);
                    words[reg][lane] = expected[reg][lane] = value;
                  }
                for (unsigned lane = 0; lane < 32; ++lane) {
                  if (!((exec_mask >> lane) & 1))
                    continue;
                  uint32_t a = words[0][lane + 1], b = words[breg][lane + 1];
                  uint32_t result = op == 8    ? a
                                    : op == 9  ? (lane & 1 ? b : a)
                                    : op == 14 ? a + b
                                    : op == 15 ? b << (a & 31)
                                               : a & b;
                  expected[target][lane + 1] = result;
                }
                const uint32_t *a = words[0] + 1, *b = words[breg] + 1;
                uint32_t *d = words[target] + 1;
                ASSERT_EQ(execute(op, cpu | semantics | GOC_SEMANTICS_STRICT, exec_mask, 0, &d, &a,
                                  &b, nullptr),
                          GOC_SUCCESS);
                EXPECT_EQ(std::memcmp(words, expected, sizeof words), 0)
                    << op << '/' << seed << '/' << target << '/' << exec_mask;
              }
}

TEST(Dual, Rdna3PairReadsPrePairSources) {
  // RDNA3 section 7.6: both slots read old values even when the other slot
  // writes the same VGPR. X: mov v0,v1; Y: mov v1,v0 satisfies bank/parity
  // restrictions. GoC executes individual slots, so the caller stages inputs.
  // This dependency is forbidden by RDNA4 section 7.8.
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (uint32_t exec_mask : exec_masks()) {
      uint32_t x[32], y[32], old_x[32], old_y[32];
      for (unsigned lane = 0; lane < 32; ++lane) {
        x[lane] = old_x[lane] = 0xa5a50000U | lane;
        y[lane] = old_y[lane] = 0x5a5a0000U | (31 - lane);
      }
      uint32_t *dx = x, *dy = y;
      const uint32_t *ax = old_y, *ay = old_x;
      uint64_t flags = cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;
      ASSERT_EQ(goc_v_dual_mov_b32(flags, exec_mask, 0, &dx, &ax), GOC_SUCCESS);
      ASSERT_EQ(goc_v_dual_mov_b32(flags, exec_mask, 0, &dy, &ay), GOC_SUCCESS);
      for (unsigned lane = 0; lane < 32; ++lane) {
        EXPECT_EQ(x[lane], (exec_mask >> lane) & 1 ? old_y[lane] : old_x[lane]);
        EXPECT_EQ(y[lane], (exec_mask >> lane) & 1 ? old_x[lane] : old_y[lane]);
      }
    }
}

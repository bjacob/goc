// SPDX-License-Identifier: MIT

#include "dpp_reference.h"
#include "fp_environment.h"
#include "gfx11_dot2.h"
#include "gfx11_dot2_fixtures.h"
#include "goc/goc.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>

namespace {

const uint64_t exact = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;

struct Registers {
  uint32_t data[4][34];
  uint32_t *v[4];

  Registers() {
    for (int i = 0; i < 4; ++i) {
      std::fill(data[i], data[i] + 34, 0xdeadbeefU);
      v[i] = data[i] + 1;
    }
  }
};

uint32_t expected(bool bf16, uint64_t mode, uint32_t a, uint32_t b, uint32_t c) {
  uint16_t a0 = a >> ((mode & GOC_DOT_LO_A_HIGH) ? 16 : 0);
  uint16_t a1 = a >> ((mode & GOC_DOT_HI_A_LOW) ? 0 : 16);
  uint16_t b0 = b >> ((mode & GOC_DOT_LO_B_HIGH) ? 16 : 0);
  uint16_t b1 = b >> ((mode & GOC_DOT_HI_B_LOW) ? 0 : 16);
  if (mode & GOC_DOT_NEG_LO_A)
    a0 ^= 0x8000;
  if (mode & GOC_DOT_NEG_HI_A)
    a1 ^= 0x8000;
  if (mode & GOC_DOT_NEG_LO_B)
    b0 ^= 0x8000;
  if (mode & GOC_DOT_NEG_HI_B)
    b1 ^= 0x8000;
  if (mode & GOC_DOT_NEG_C)
    c ^= 0x80000000;
  return bf16 ? goc::gfx11_dot2_f32<true>(a0, b0, a1, b1, c)
              : goc::gfx11_dot2_f32<false>(a0, b0, a1, b1, c);
}

} // namespace

TEST(Dot2Architecture, Gfx11HardwareCapturesMasksAliasesAndHostState) {
  goc_test::ScopedFpEnvironment restore;
  ASSERT_TRUE(restore.saved());
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    ASSERT_EQ(std::fesetround(rounding), 0);
    ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
    ASSERT_EQ(std::feraiseexcept(FE_INEXACT), 0);
    for (bool bf16 : {false, true}) {
      auto fn = bf16 ? goc_v_dot2_f32_bf16 : goc_v_dot2_f32_f16;
      const auto *cases = bf16 ? kDot2BF16Cases.data() : kDot2F16Cases.data();
      size_t count = bf16 ? kDot2BF16Cases.size() : kDot2F16Cases.size();
      for (size_t base = 0; base < count; base += 32)
        for (uint32_t exec_mask : {0U, UINT32_MAX, 0x80018001U, 0xaaaaaaaaU})
          for (int dst = 0; dst < 4; ++dst) {
            Registers r;
            for (int lane = 0; lane < 32; ++lane) {
              auto f = cases[(base + lane) % count];
              r.v[0][lane] = f.a;
              r.v[1][lane] = f.b;
              r.v[2][lane] = f.c;
            }
            uint32_t wanted[4][34];
            std::memcpy(wanted, r.data, sizeof wanted);
            for (int lane = 0; lane < 32; ++lane)
              if ((exec_mask >> lane) & 1)
                wanted[dst][lane + 1] = cases[(base + lane) % count].expected;
            ASSERT_EQ(fn(exact, exec_mask, 0, r.v + dst, r.v, r.v + 1, r.v + 2, nullptr),
                      GOC_SUCCESS);
            EXPECT_EQ(std::memcmp(wanted, r.data, sizeof wanted), 0);
          }
    }
    EXPECT_EQ(std::fegetround(), rounding);
    EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_INEXACT);
  }
}

TEST(Dot2Architecture, Gfx11ModifiersDppMasksAndAliases) {
  for (bool bf16 : {false, true}) {
    auto fn = bf16 ? goc_v_dot2_f32_bf16 : goc_v_dot2_f32_f16;
    const auto *cases = bf16 ? kDot2BF16Cases.data() : kDot2F16Cases.data();
    for (unsigned selections = 0; selections < 16; ++selections)
      for (unsigned signs = 0; signs < 32; ++signs)
        for (uint64_t routing : goc_test::dpp_modes)
          for (int dst = 0; dst < 4; ++dst) {
            uint64_t mode = routing | (selections << 7) | signs | GOC_DOT_CLAMP;
            uint32_t exec_mask = 0xd7efb579U;
            Registers r;
            for (int lane = 0; lane < 32; ++lane) {
              r.v[0][lane] = cases[lane].a;
              r.v[1][lane] = cases[lane].b;
              r.v[2][lane] = cases[lane].c;
            }
            uint32_t wanted[4][34];
            std::memcpy(wanted, r.data, sizeof wanted);
            for (int lane = 0; lane < 32; ++lane) {
              int source;
              if (goc_test::dpp_source(routing, exec_mask, lane, source))
                wanted[dst][lane + 1] = expected(bf16, mode, source < 0 ? 0 : r.v[0][source],
                                                 r.v[1][lane], r.v[2][lane]);
            }
            ASSERT_EQ(fn(exact, exec_mask, mode, r.v + dst, r.v, r.v + 1, r.v + 2, nullptr),
                      GOC_SUCCESS);
            EXPECT_EQ(std::memcmp(wanted, r.data, sizeof wanted), 0);
          }
  }
}

TEST(Dot2Architecture, DifferentExactResultsAndOptionalReporting) {
  Registers r;
  for (int lane = 0; lane < 32; ++lane) {
    r.v[0][lane] = kDot2F16Cases[0].a;
    r.v[1][lane] = kDot2F16Cases[0].b;
    r.v[2][lane] = kDot2F16Cases[0].c;
  }
  ASSERT_EQ(goc_v_dot2_f32_f16(exact, UINT32_MAX, 0, r.v + 3, r.v, r.v + 1, r.v + 2, nullptr),
            GOC_SUCCESS);
  uint32_t rdna3 = r.v[3][0];
  ASSERT_EQ(goc_v_dot2_f32_f16_rdna4(exact, UINT32_MAX, 0, r.v + 3, r.v, r.v + 1, r.v + 2, nullptr),
            GOC_SUCCESS);
  EXPECT_EQ(rdna3, 0x425c0001U);
  EXPECT_NE(rdna3, r.v[3][0]);
  for (auto fn : {goc_v_dot2_f32_f16, goc_v_dot2_f32_bf16}) {
    uint32_t state = 0x80000055;
    EXPECT_EQ(fn(exact, UINT32_MAX, 0, nullptr, nullptr, nullptr, nullptr, &state),
              GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
    EXPECT_EQ(state, 0x80000055U);
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      EXPECT_EQ(fn(cpu, UINT32_MAX, 0, r.v + 3, r.v, r.v + 1, r.v + 2, &state), GOC_SUCCESS);
      EXPECT_EQ(state, 0x80000055U);
    }
    for (unsigned bit : {5U, 11U, 31U, 63U})
      EXPECT_EQ(fn(exact, UINT32_MAX, 1ULL << bit, nullptr, nullptr, nullptr, nullptr, nullptr),
                GOC_ERROR_INVALID_FLAGS);
  }
}

TEST(Dot2Architecture, AccumulateHardwareAndDppAliases) {
  for (size_t base = 0; base < kDot2F16Cases.size(); base += 32)
    for (uint32_t exec_mask : {0U, UINT32_MAX, 0x80018001U, 0xaaaaaaaaU}) {
      Registers r;
      for (int lane = 0; lane < 32; ++lane) {
        const auto &f = kDot2F16Cases[(base + lane) % kDot2F16Cases.size()];
        r.v[0][lane] = f.a;
        r.v[1][lane] = f.b;
        r.v[2][lane] = f.c;
      }
      ASSERT_EQ(goc_v_dot2acc_f32_f16(exact, exec_mask, 0, r.v + 2, r.v, r.v + 1, nullptr),
                GOC_SUCCESS);
      for (int lane = 0; lane < 32; ++lane) {
        const auto &f = kDot2F16Cases[(base + lane) % kDot2F16Cases.size()];
        EXPECT_EQ(r.v[2][lane], ((exec_mask >> lane) & 1) ? f.expected : f.c);
      }
    }
  for (auto mode : goc_test::dpp_modes)
    for (int dst : {0, 1, 2}) {
      Registers r;
      uint32_t wanted[4][34];
      for (int lane = 0; lane < 32; ++lane) {
        r.v[0][lane] = kDot2F16Cases[lane].a;
        r.v[1][lane] = kDot2F16Cases[lane].b;
        r.v[2][lane] = kDot2F16Cases[lane].c;
      }
      std::memcpy(wanted, r.data, sizeof wanted);
      uint32_t exec_mask = 0xd7efb579U;
      for (int lane = 0; lane < 32; ++lane) {
        int source;
        if (goc_test::dpp_source(mode, exec_mask, lane, source))
          wanted[dst][lane + 1] =
              expected(false, 0, source < 0 ? 0 : r.v[0][source], r.v[1][lane], r.v[dst][lane]);
      }
      ASSERT_EQ(goc_v_dot2acc_f32_f16(exact, exec_mask, mode, r.v + dst, r.v, r.v + 1, nullptr),
                GOC_SUCCESS);
      EXPECT_EQ(std::memcmp(wanted, r.data, sizeof wanted), 0);
    }
  for (unsigned bit = 0; bit < 32; ++bit)
    EXPECT_EQ(
        goc_v_dot2acc_f32_f16(exact, UINT32_MAX, 1ULL << bit, nullptr, nullptr, nullptr, nullptr),
        GOC_ERROR_INVALID_FLAGS);
}

TEST(Dot2Architecture, AccumulateLooseSimdAndReportingOptOut) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    Registers r;
    for (int lane = 0; lane < 32; ++lane) {
      r.v[0][lane] = r.v[1][lane] = 0x3c003c00;
      r.v[2][lane] = 0x3f800000;
    }
    uint32_t state = 0x80000055;
    ASSERT_EQ(goc_v_dot2acc_f32_f16(cpu, 0xaaaaaaaaU, 0, r.v + 2, r.v, r.v + 1, &state),
              GOC_SUCCESS);
    for (int lane = 0; lane < 32; ++lane)
      EXPECT_EQ(r.v[2][lane], (lane & 1) ? 0x40400000U : 0x3f800000U);
    EXPECT_EQ(state, 0x80000055U);
    EXPECT_EQ(goc_v_dot2acc_f32_f16(cpu | exact, UINT32_MAX, 0, nullptr, nullptr, nullptr, &state),
              GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
    EXPECT_EQ(state, 0x80000055U);
  }
}

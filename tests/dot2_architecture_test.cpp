// SPDX-License-Identifier: MIT

#include "dpp_reference.h"
#include "fp_environment.h"
#include "fuzzy.h"
#include "gfx11_dot2.h"
#include "gfx11_dot2_fixtures.h"
#include "goc/goc.h"
#include "loose_tolerances.h"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {

const uint64_t exact = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;

template <unsigned Lanes> struct LaneRegisters {
  uint32_t data[4][Lanes + 2];
  uint32_t *v[4];

  LaneRegisters() {
    for (int i = 0; i < 4; ++i) {
      std::fill(data[i], data[i] + Lanes + 2, 0xdeadbeefU);
      v[i] = data[i] + 1;
    }
  }
};

using Registers = LaneRegisters<32>;
using WideRegisters = LaneRegisters<64>;

// Modifier/routing reference reuses the borrowed model; hardware fixtures below
// independently anchor its unmodified arithmetic, not every transformed input.
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
  for (bool bf16 : {false, true}) {
    const auto *cases = bf16 ? kDot2BF16Cases.data() : kDot2F16Cases.data();
    size_t count = bf16 ? kDot2BF16Cases.size() : kDot2F16Cases.size();
    for (size_t i = 0; i < count; ++i)
      EXPECT_EQ(expected(bf16, 0, cases[i].a, cases[i].b, cases[i].c), cases[i].expected)
          << "bf16=" << bf16 << " capture=" << i;
  }

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
            EXPECT_EQ(std::memcmp(wanted, r.data, sizeof wanted), 0) << "destination=" << dst;
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
            EXPECT_EQ(std::memcmp(wanted, r.data, sizeof wanted), 0) << "destination=" << dst;
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
      EXPECT_EQ(std::memcmp(wanted, r.data, sizeof wanted), 0) << "destination=" << dst;
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

TEST(Dot2Architecture, Wave64Gfx11HardwareCapturesMasksAliasesAndHostState) {
  goc_test::ScopedFpEnvironment restore;
  ASSERT_TRUE(restore.saved());
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    ASSERT_EQ(std::fesetround(rounding), 0);
    ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
    ASSERT_EQ(std::feraiseexcept(FE_INEXACT), 0);
    for (bool bf16 : {false, true}) {
      auto fn = bf16 ? goc_v_dot2_f32_bf16_wave64 : goc_v_dot2_f32_f16_wave64;
      const auto *cases = bf16 ? kDot2BF16Cases.data() : kDot2F16Cases.data();
      size_t count = bf16 ? kDot2BF16Cases.size() : kDot2F16Cases.size();
      for (size_t base = 0; base < count; base += 64)
        for (uint64_t exec_mask :
             {uint64_t(0), UINT64_MAX, uint64_t(0x8001800180018001ULL),
              uint64_t(0xffffffff00000000ULL), uint64_t(0xaaaaaaaaaaaaaaaaULL)})
          for (int dst = 0; dst < 4; ++dst) {
            WideRegisters r;
            for (int lane = 0; lane < 64; ++lane) {
              auto f = cases[(base + lane) % count];
              r.v[0][lane] = f.a;
              r.v[1][lane] = f.b;
              r.v[2][lane] = f.c;
            }
            uint32_t wanted[4][66];
            std::memcpy(wanted, r.data, sizeof wanted);
            for (int lane = 0; lane < 64; ++lane)
              if ((exec_mask >> lane) & 1)
                wanted[dst][lane + 1] = cases[(base + lane) % count].expected;
            ASSERT_EQ(fn(exact, exec_mask, 0, r.v + dst, r.v, r.v + 1, r.v + 2, nullptr),
                      GOC_SUCCESS);
            EXPECT_EQ(std::memcmp(wanted, r.data, sizeof wanted), 0) << "destination=" << dst;
          }
    }
    EXPECT_EQ(std::fegetround(), rounding);
    EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_INEXACT);
  }
}

TEST(Dot2Architecture, Wave64Gfx11ModifiersDppMasksAndAliases) {
  for (bool bf16 : {false, true}) {
    auto fn = bf16 ? goc_v_dot2_f32_bf16_wave64 : goc_v_dot2_f32_f16_wave64;
    const auto *cases = bf16 ? kDot2BF16Cases.data() : kDot2F16Cases.data();
    for (unsigned selections = 0; selections < 16; ++selections)
      for (unsigned signs = 0; signs < 32; ++signs)
        for (uint64_t routing : goc_test::dpp_modes)
          for (int dst = 0; dst < 4; ++dst) {
            uint64_t mode = routing | (selections << 7) | signs | GOC_DOT_CLAMP;
            uint64_t exec_mask = 0x97efd7efd7efb579ULL;
            WideRegisters r;
            for (int lane = 0; lane < 64; ++lane) {
              r.v[0][lane] = cases[lane % 32].a;
              r.v[1][lane] = cases[lane % 32].b;
              r.v[2][lane] = cases[lane % 32].c;
            }
            uint32_t wanted[4][66];
            std::memcpy(wanted, r.data, sizeof wanted);
            for (int lane = 0; lane < 64; ++lane) {
              int source;
              if (goc_test::dpp_source(routing, exec_mask, lane, source))
                wanted[dst][lane + 1] = expected(bf16, mode, source < 0 ? 0 : r.v[0][source],
                                                 r.v[1][lane], r.v[2][lane]);
            }
            ASSERT_EQ(fn(exact, exec_mask, mode, r.v + dst, r.v, r.v + 1, r.v + 2, nullptr),
                      GOC_SUCCESS);
            EXPECT_EQ(std::memcmp(wanted, r.data, sizeof wanted), 0) << "destination=" << dst;
          }
  }
}

TEST(Dot2Architecture, Wave64DifferentExactResultsAndOptionalReporting) {
  WideRegisters r;
  for (int lane = 0; lane < 64; ++lane) {
    r.v[0][lane] = kDot2F16Cases[0].a;
    r.v[1][lane] = kDot2F16Cases[0].b;
    r.v[2][lane] = kDot2F16Cases[0].c;
  }
  ASSERT_EQ(
      goc_v_dot2_f32_f16_wave64(exact, UINT64_MAX, 0, r.v + 3, r.v, r.v + 1, r.v + 2, nullptr),
      GOC_SUCCESS);
  uint32_t rdna3 = r.v[3][0];
  ASSERT_EQ(goc_v_dot2_f32_f16_rdna4_wave64(exact, UINT64_MAX, 0, r.v + 3, r.v, r.v + 1, r.v + 2,
                                            nullptr),
            GOC_SUCCESS);
  EXPECT_EQ(rdna3, 0x425c0001U);
  EXPECT_NE(rdna3, r.v[3][0]);
  for (auto fn : {goc_v_dot2_f32_f16_wave64, goc_v_dot2_f32_bf16_wave64}) {
    uint32_t state = 0x80000055;
    EXPECT_EQ(fn(exact, UINT64_MAX, 0, nullptr, nullptr, nullptr, nullptr, &state),
              GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
    EXPECT_EQ(state, 0x80000055U);
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      EXPECT_EQ(fn(cpu, UINT64_MAX, 0, r.v + 3, r.v, r.v + 1, r.v + 2, &state), GOC_SUCCESS);
      EXPECT_EQ(state, 0x80000055U);
    }
    for (unsigned bit : {5U, 11U, 31U, 63U})
      EXPECT_EQ(fn(exact, UINT64_MAX, 1ULL << bit, nullptr, nullptr, nullptr, nullptr, nullptr),
                GOC_ERROR_INVALID_FLAGS);
  }
}

TEST(Dot2Architecture, Wave64AccumulateHardwareAndDppAliases) {
  for (size_t base = 0; base < kDot2F16Cases.size(); base += 64)
    for (uint64_t exec_mask : {uint64_t(0), UINT64_MAX, uint64_t(0x8001800180018001ULL),
                               uint64_t(0xffffffff00000000ULL), uint64_t(0xaaaaaaaaaaaaaaaaULL)}) {
      WideRegisters r;
      for (int lane = 0; lane < 64; ++lane) {
        const auto &f = kDot2F16Cases[(base + lane) % kDot2F16Cases.size()];
        r.v[0][lane] = f.a;
        r.v[1][lane] = f.b;
        r.v[2][lane] = f.c;
      }
      ASSERT_EQ(goc_v_dot2acc_f32_f16_wave64(exact, exec_mask, 0, r.v + 2, r.v, r.v + 1, nullptr),
                GOC_SUCCESS);
      for (int lane = 0; lane < 64; ++lane) {
        const auto &f = kDot2F16Cases[(base + lane) % kDot2F16Cases.size()];
        EXPECT_EQ(r.v[2][lane], ((exec_mask >> lane) & 1) ? f.expected : f.c);
      }
    }
  for (auto mode : goc_test::dpp_modes)
    for (int dst : {0, 1, 2}) {
      WideRegisters r;
      uint32_t wanted[4][66];
      for (int lane = 0; lane < 64; ++lane) {
        r.v[0][lane] = kDot2F16Cases[lane % kDot2F16Cases.size()].a;
        r.v[1][lane] = kDot2F16Cases[lane % kDot2F16Cases.size()].b;
        r.v[2][lane] = kDot2F16Cases[lane % kDot2F16Cases.size()].c;
      }
      std::memcpy(wanted, r.data, sizeof wanted);
      uint64_t exec_mask = 0x97efd7efd7efb579ULL;
      for (int lane = 0; lane < 64; ++lane) {
        int source;
        if (goc_test::dpp_source(mode, exec_mask, lane, source))
          wanted[dst][lane + 1] =
              expected(false, 0, source < 0 ? 0 : r.v[0][source], r.v[1][lane], r.v[dst][lane]);
      }
      ASSERT_EQ(
          goc_v_dot2acc_f32_f16_wave64(exact, exec_mask, mode, r.v + dst, r.v, r.v + 1, nullptr),
          GOC_SUCCESS);
      EXPECT_EQ(std::memcmp(wanted, r.data, sizeof wanted), 0) << "destination=" << dst;
    }
  for (unsigned bit = 0; bit < 32; ++bit)
    EXPECT_EQ(goc_v_dot2acc_f32_f16_wave64(exact, UINT64_MAX, 1ULL << bit, nullptr, nullptr,
                                           nullptr, nullptr),
              GOC_ERROR_INVALID_FLAGS);
}

TEST(Dot2Architecture, Wave64AccumulateLooseSimdAndReportingOptOut) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    WideRegisters r;
    for (int lane = 0; lane < 64; ++lane) {
      r.v[0][lane] = r.v[1][lane] = 0x3c003c00;
      r.v[2][lane] = 0x3f800000;
    }
    uint32_t state = 0x80000055;
    ASSERT_EQ(
        goc_v_dot2acc_f32_f16_wave64(cpu, 0xaaaaaaaaaaaaaaaaULL, 0, r.v + 2, r.v, r.v + 1, &state),
        GOC_SUCCESS);
    for (int lane = 0; lane < 64; ++lane)
      EXPECT_EQ(r.v[2][lane], (lane & 1) ? 0x40400000U : 0x3f800000U);
    EXPECT_EQ(state, 0x80000055U);
    EXPECT_EQ(
        goc_v_dot2acc_f32_f16_wave64(cpu | exact, UINT64_MAX, 0, nullptr, nullptr, nullptr, &state),
        GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
    EXPECT_EQ(state, 0x80000055U);
  }
}

TEST(Dot2Architecture, AccumulateStateReportingPrecedesFlagValidation) {
  for (bool wide : {false, true})
    for (bool strict : {false, true})
      for (uint64_t semantics : {GOC_SEMANTICS_EXACT_EMPIRICAL, uint64_t(2) << 16})
        for (uint64_t exec_mask : {uint64_t(0), uint64_t(1), UINT64_MAX})
          for (unsigned bit = 0; bit < 64; ++bit) {
            uint32_t words[3][64], before[3][64], state = 0x8badf00d;
            for (unsigned reg = 0; reg < 3; ++reg)
              for (unsigned lane = 0; lane < 64; ++lane)
                words[reg][lane] = 0x12340000U + 64 * reg + lane;
            std::memcpy(before, words, sizeof words);
            const uint64_t flags = semantics | (strict ? GOC_SEMANTICS_STRICT : 0);
            const uint64_t mode = 1ULL << bit;
            auto call = [&](uint64_t f, bool null_operands, uint32_t *state_output) {
              uint32_t *d = words[2];
              const uint32_t *a = words[0], *b = words[1];
              return wide ? goc_v_dot2acc_f32_f16_wave64(f, exec_mask, mode,
                                                         null_operands ? nullptr : &d,
                                                         null_operands ? nullptr : &a,
                                                         null_operands ? nullptr : &b, state_output)
                          : goc_v_dot2acc_f32_f16(f, uint32_t(exec_mask), mode,
                                                  null_operands ? nullptr : &d,
                                                  null_operands ? nullptr : &a,
                                                  null_operands ? nullptr : &b, state_output);
            };
            for (bool null_operands : {false, true})
              EXPECT_EQ(call(flags, null_operands, &state), GOC_ERROR_UNSUPPORTED_GLOBAL_STATE)
                  << wide << '/' << strict << '/' << bit;
            // Low bits are all invalid for this VOP2 form. Opting out of
            // reporting, or requesting loose semantics, retains that error.
            if (bit < 32) {
              EXPECT_EQ(call(flags, false, nullptr), GOC_ERROR_INVALID_FLAGS);
              EXPECT_EQ(call(0, false, &state), GOC_ERROR_INVALID_FLAGS);
            }
            EXPECT_EQ(state, 0x8badf00dU);
            EXPECT_EQ(std::memcmp(words, before, sizeof words), 0);
          }
}

// Normal factors bounded by four avoid overflow/flush ambiguities. Independent
// double arithmetic covers loose numerical behavior; this does not claim that
// RDNA3's empirical exact accumulator equals the mathematical sum.
TEST(Dot2Architecture, LooseFractionalInputsModifiersAndStorage) {
  for (bool bf16 : {false, true})
    for (bool wide : {false, true})
      for (bool accumulate : {false, true}) {
        if (bf16 && accumulate)
          continue;
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
          std::mt19937 random(42683);
          for (unsigned batch = 0; batch < 128; ++batch) {
            WideRegisters r;
            double factors[2][64][2], sums[64];
            for (unsigned reg = 0; reg < 2; ++reg)
              for (unsigned lane = 0; lane < 64; ++lane) {
                r.v[reg][lane] = 0;
                for (unsigned half = 0; half < 2; ++half) {
                  unsigned fraction_bits = bf16 ? 7 : 10, bias = bf16 ? 127 : 15;
                  uint32_t draw = random();
                  unsigned fraction = draw & ((1U << fraction_bits) - 1);
                  int exponent = int((draw >> 12) % 10) - 8;
                  bool negative = draw >> 31;
                  uint16_t code =
                      (negative ? 0x8000 : 0) | ((exponent + bias) << fraction_bits) | fraction;
                  r.v[reg][lane] |= uint32_t(code) << (16 * half);
                  factors[reg][lane][half] = std::ldexp(double((1U << fraction_bits) + fraction),
                                                        exponent - int(fraction_bits)) *
                                             (negative ? -1 : 1);
                }
              }
            for (unsigned lane = 0; lane < 64; ++lane) {
              float c = float(int(random() % 2049) - 1024) / 257;
              std::memcpy(&r.v[2][lane], &c, sizeof(c));
              sums[lane] = c;
            }
            uint32_t signs = random() & 31;
            uint32_t selection = random() & 15;
            uint32_t mode =
                accumulate ? 0 : signs | (selection << 7) | (batch & 1 ? GOC_DOT_CLAMP : 0);
            unsigned dst = accumulate ? 2 : batch % 4;
            uint64_t exec_mask = batch & 1 ? 0x87654321fedcba98ULL : UINT64_MAX;
            uint32_t before[4][66];
            std::memcpy(before, r.data, sizeof(before));
            uint32_t state = 0xffffff80;
            int status;
            if (accumulate)
              status = wide ? goc_v_dot2acc_f32_f16_wave64(cpu, exec_mask, 0, r.v + dst, r.v,
                                                           r.v + 1, &state)
                            : goc_v_dot2acc_f32_f16(cpu, uint32_t(exec_mask), 0, r.v + dst, r.v,
                                                    r.v + 1, &state);
            else if (wide)
              status = (bf16 ? goc_v_dot2_f32_bf16_wave64 : goc_v_dot2_f32_f16_wave64)(
                  cpu, exec_mask, mode, r.v + dst, r.v, r.v + 1, r.v + 2, &state);
            else
              status = (bf16 ? goc_v_dot2_f32_bf16 : goc_v_dot2_f32_f16)(
                  cpu, uint32_t(exec_mask), mode, r.v + dst, r.v, r.v + 1, r.v + 2, &state);
            ASSERT_EQ(status, GOC_SUCCESS);
            ASSERT_EQ(state, 0xffffff80U);
            for (unsigned reg = 0; reg < 4; ++reg)
              for (unsigned word = 0; word < 66; ++word) {
                bool active = reg == dst && word > 0 && word <= (wide ? 64U : 32U) &&
                              ((exec_mask >> (word - 1)) & 1);
                if (!active) {
                  ASSERT_EQ(r.data[reg][word], before[reg][word]) << reg << "/" << word;
                  continue;
                }
                unsigned lane = word - 1;
                double a0 = factors[0][lane][bool(mode & GOC_DOT_LO_A_HIGH)];
                double a1 = factors[0][lane][!bool(mode & GOC_DOT_HI_A_LOW)];
                double b0 = factors[1][lane][bool(mode & GOC_DOT_LO_B_HIGH)];
                double b1 = factors[1][lane][!bool(mode & GOC_DOT_HI_B_LOW)];
                if (mode & GOC_DOT_NEG_LO_A)
                  a0 = -a0;
                if (mode & GOC_DOT_NEG_HI_A)
                  a1 = -a1;
                if (mode & GOC_DOT_NEG_LO_B)
                  b0 = -b0;
                if (mode & GOC_DOT_NEG_HI_B)
                  b1 = -b1;
                double c = (mode & GOC_DOT_NEG_C) ? -sums[lane] : sums[lane];
                float actual;
                std::memcpy(&actual, &r.data[reg][word], sizeof(actual));
                ASSERT_TRUE(goc_test::fuzzy(actual, a0 * b0 + a1 * b1 + c,
                                            goc_test::dot2_bounded_tolerance))
                    << "bf16=" << bf16 << " wide=" << wide << " accumulate=" << accumulate
                    << " cpu=" << cpu << " batch=" << batch << " mode=" << mode << " lane=" << lane;
              }
          }
        }
      }
}

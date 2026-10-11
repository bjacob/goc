// SPDX-License-Identifier: MIT

#include "fp_environment.h"
#include "fuzzy.h"
#include "gfx11_dot2.h"
#include "gfx11_dot2_fixtures.h"
#include "goc/goc.h"
#include "internal.h"
#include "loose_tolerances.h"
#include "matrix_capture_state.h"
#include "packed16.h"
#include "packed_wmma_fixtures.h"
#include "rdna_wmma_bf16_bf16_cases.h"
#include "rdna_wmma_f16_f16_cases.h"
#include "rdna_wmma_f32_bf16_cases.h"
#include "rdna_wmma_f32_f16_cases.h"
#include "rdna_wmma_iu4_cases.h"
#include "rdna_wmma_iu4_k32_cases.h"
#include "rdna_wmma_iu8_cases.h"
#include "wmma_replicated_packed_fixtures.h"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>

namespace {

using Wmma = decltype(&goc_v_wmma_f32_16x16x16_f16);

Wmma floating(bool bf16, bool packed, int lanes, bool rdna4 = false) {
  if (rdna4) {
    if (packed) {
      if (lanes == 64)
        return bf16 ? goc_v_wmma_bf16_16x16x16_bf16_rdna4_wave64
                    : goc_v_wmma_f16_16x16x16_f16_rdna4_wave64;
      return bf16 ? goc_v_wmma_bf16_16x16x16_bf16_rdna4 : goc_v_wmma_f16_16x16x16_f16_rdna4;
    }
    if (lanes == 64)
      return bf16 ? goc_v_wmma_f32_16x16x16_bf16_rdna4_wave64
                  : goc_v_wmma_f32_16x16x16_f16_rdna4_wave64;
    return bf16 ? goc_v_wmma_f32_16x16x16_bf16_rdna4 : goc_v_wmma_f32_16x16x16_f16_rdna4;
  }
  if (packed) {
    if (lanes == 64)
      return bf16 ? goc_v_wmma_bf16_16x16x16_bf16_wave64 : goc_v_wmma_f16_16x16x16_f16_wave64;
    return bf16 ? goc_v_wmma_bf16_16x16x16_bf16 : goc_v_wmma_f16_16x16x16_f16;
  }
  if (lanes == 64)
    return bf16 ? goc_v_wmma_f32_16x16x16_bf16_wave64 : goc_v_wmma_f32_16x16x16_f16_wave64;
  return bf16 ? goc_v_wmma_f32_16x16x16_bf16 : goc_v_wmma_f32_16x16x16_f16;
}

struct Registers {
  uint32_t data[32][66];
  uint32_t *v[32];

  Registers() {
    for (int r = 0; r < 32; ++r) {
      std::fill(data[r], data[r] + 66, 0xa55a1234U ^ (r * 0x10201U));
      v[r] = data[31 - r] + 1;
    }
  }
};

const uint64_t exact = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;

} // namespace

TEST(WmmaReplicatedExact, BorrowedDot2HardwareBitsUnderHostFpModes) {
  goc_test::ScopedFpEnvironment restore;
  ASSERT_TRUE(restore.saved());
  for (int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
    ASSERT_EQ(std::fesetround(rounding), 0);
    ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
    ASSERT_EQ(std::feraiseexcept(FE_INEXACT), 0);
    for (const auto &f : kDot2F16Cases)
      EXPECT_EQ(goc::gfx11_dot2_f32<false>(f.a, f.b, f.a >> 16, f.b >> 16, f.c), f.expected);
    for (const auto &f : kDot2BF16Cases)
      EXPECT_EQ(goc::gfx11_dot2_f32<true>(f.a, f.b, f.a >> 16, f.b >> 16, f.c), f.expected);
    EXPECT_EQ(std::fegetround(), rounding);
    EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_INEXACT);
  }
}

TEST(WmmaReplicatedExact, BorrowedFp32HardwareMatrix) {
  // Same input generator as rocjitsu Gfx11Dot2.WmmaHardwareOrderAndOverlappingAccumulator.
  for (int lanes : {32, 64})
    for (bool bf16 : {false, true})
      for (int dst : {0, 8, 16, 24})
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
          Registers r;
          auto bits16 = [bf16](int value) {
            uint32_t bits = goc::as_bits(float(value));
            if (bf16)
              return uint16_t(bits >> 16);
            return goc::packed16::pack_f16(value, 0);
          };
          for (int lane = 0; lane < lanes; ++lane) {
            int index = lane % 16;
            for (int reg = 0; reg < 8; ++reg) {
              int a0 = ((index * 16 + reg * 2) * 7) % 17 - 8;
              int a1 = ((index * 16 + reg * 2 + 1) * 7) % 17 - 8;
              int b0 = ((reg * 2 * 16 + index) * 11) % 17 - 8;
              int b1 = (((reg * 2 + 1) * 16 + index) * 11) % 17 - 8;
              r.v[reg][lane] = bits16(a0) | uint32_t(bits16(a1)) << 16;
              r.v[8 + reg][lane] = bits16(b0) | uint32_t(bits16(b1)) << 16;
            }
            for (int reg = 0; reg < 256 / lanes; ++reg) {
              int c = (((lanes / 16 * reg + lane / 16) * 16 + index) * 13) % 129 - 64;
              r.v[16 + reg][lane] = goc::as_bits(float(c));
            }
          }
          uint32_t before[32][66];
          std::memcpy(before, r.data, sizeof(before));
          ASSERT_EQ(floating(bf16, false, lanes)(cpu | exact, 0, r.v + dst, r.v, r.v + 8, r.v + 16),
                    GOC_SUCCESS);
          for (int reg = 0; reg < 256 / lanes; ++reg)
            for (int lane = 0; lane < lanes; ++lane)
              EXPECT_EQ(r.v[dst + reg][lane],
                        kWmmaIntegerResult[(reg * (lanes / 16) + lane / 16) * 16 + lane % 16]);
          for (int reg = 0; reg < 32; ++reg)
            for (int word = 0; word < 66; ++word)
              if (reg < dst || reg >= dst + 256 / lanes || word == 0 || word > lanes) {
                EXPECT_EQ(r.data[31 - reg][word], before[31 - reg][word]);
              }
        }
}

TEST(WmmaReplicatedExact, PackedHardwareMatricesBothHalvesAliasesAndHostState) {
  goc_test::ScopedFpEnvironment restore;
  ASSERT_TRUE(restore.saved());
  for (int lanes : {32, 64})
    for (bool bf16 : {false, true})
      for (int fixture = 0; fixture < 7; ++fixture)
        for (unsigned high : {0U, 1U})
          for (int dst : {0, 4, 8, 16, 24})
            for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
              for (int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
                Registers r;
                const auto &input = kPackedInputs[bf16][fixture];
                for (int lane = 0; lane < lanes; ++lane) {
                  int index = lane % 16;
                  for (int reg = 0; reg < 8; ++reg) {
                    r.v[reg][lane] = input.a[index * 16 + 2 * reg] |
                                     (uint32_t(input.a[index * 16 + 2 * reg + 1]) << 16);
                    r.v[8 + reg][lane] = input.b[2 * reg * 16 + index] |
                                         (uint32_t(input.b[(2 * reg + 1) * 16 + index]) << 16);
                  }
                  for (int reg = 0; reg < 256 / lanes; ++reg) {
                    int row = (lanes / 16) * reg + lane / 16;
                    r.v[16 + reg][lane] = (uint32_t(input.c[row * 16 + index]) << (16 * high)) |
                                          (0x1234U << (16 * (1 - high)));
                  }
                }
                uint32_t expected[32][66];
                std::memcpy(expected, r.data, sizeof(expected));
                for (int reg = 0; reg < 256 / lanes; ++reg)
                  for (int lane = 0; lane < lanes; ++lane) {
                    int row = reg * (lanes / 16) + lane / 16;
                    uint32_t value =
                        kReplicatedPackedExpected[lanes == 64][bf16][fixture][row * 16 + lane % 16];
                    auto &word = expected[31 - dst - reg][lane + 1];
                    word = (word & ~(65535U << (16 * high))) | (value << (16 * high));
                  }
                ASSERT_EQ(std::fesetround(rounding), 0);
                ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
                ASSERT_EQ(std::feraiseexcept(FE_INEXACT), 0);
                ASSERT_EQ(floating(bf16, true, lanes)(cpu | exact, high ? GOC_WMMA_HIGH_C_D : 0,
                                                      r.v + dst, r.v, r.v + 8, r.v + 16),
                          GOC_SUCCESS);
                EXPECT_EQ(std::memcmp(expected, r.data, sizeof(expected)), 0)
                    << lanes << "/" << bf16 << "/" << fixture;
                EXPECT_EQ(std::fegetround(), rounding);
                EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_INEXACT);
              }
}

TEST(WmmaReplicatedExact, EveryFloatingModifierAndHalfSelection) {
  // Apply modifiers once through the API and once by transforming raw sources.
  // Hardware fixtures above independently anchor the arithmetic model.
  for (int lanes : {32, 64})
    for (bool bf16 : {false, true})
      for (bool packed : {false, true})
        for (unsigned high = 0; high < (packed ? 2U : 1U); ++high)
          for (uint64_t mode = 0; mode < 64; ++mode) {
            Registers raw, transformed;
            for (int lane = 0; lane < lanes; ++lane) {
              for (int reg = 0; reg < 8; ++reg) {
                unsigned index = (lane % 16) * 8 + reg;
                uint32_t av = index * 0x1f123bb5U, bv = index * 0x9e3779b9U;
                raw.v[reg][lane] = av;
                raw.v[8 + reg][lane] = bv;
                transformed.v[reg][lane] =
                    av ^ ((mode & 1) ? 0x8000U : 0) ^ ((mode & 8) ? 0x80000000U : 0);
                transformed.v[8 + reg][lane] =
                    bv ^ ((mode & 2) ? 0x8000U : 0) ^ ((mode & 16) ? 0x80000000U : 0);
              }
              for (int reg = 0; reg < 256 / lanes; ++reg) {
                uint32_t cv = (lane + reg * lanes) * 0x6c078965U;
                raw.v[16 + reg][lane] = cv;
                uint32_t sign = packed ? 0x8000U << (16 * high) : 0x80000000U;
                if (mode & 32)
                  cv &= ~sign;
                if (mode & 4)
                  cv ^= sign;
                transformed.v[16 + reg][lane] = cv;
              }
            }
            uint64_t half = high ? GOC_WMMA_HIGH_C_D : 0;
            auto fn = floating(bf16, packed, lanes);
            ASSERT_EQ(fn(exact, mode | half, raw.v + 24, raw.v, raw.v + 8, raw.v + 16),
                      GOC_SUCCESS);
            ASSERT_EQ(fn(exact, half, transformed.v + 24, transformed.v, transformed.v + 8,
                         transformed.v + 16),
                      GOC_SUCCESS);
            for (int reg = 24; reg < 24 + 256 / lanes; ++reg)
              for (int lane = 0; lane < lanes; ++lane)
                EXPECT_EQ(raw.v[reg][lane], transformed.v[reg][lane])
                    << lanes << "/" << packed << "/" << mode;
          }
}

TEST(WmmaReplicatedExact, PackedOverflowAndValidation) {
  for (int lanes : {32, 64})
    for (bool bf16 : {false, true}) {
      Registers r;
      auto fn = floating(bf16, true, lanes);
      uint32_t before[32][66];
      std::memcpy(before, r.data, sizeof(before));
      for (unsigned bit = 0; bit < 64; ++bit) {
        if ((1ULL << bit) & (63ULL | GOC_WMMA_HIGH_C_D))
          continue;
        EXPECT_EQ(fn(exact, 1ULL << bit, r.v + 24, r.v, r.v + 8, r.v + 16),
                  GOC_ERROR_INVALID_FLAGS);
      }
      EXPECT_EQ(fn(GOC_SEMANTICS_MASK | GOC_SEMANTICS_STRICT, 0, r.v + 24, r.v, r.v + 8, r.v + 16),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
      EXPECT_EQ(std::memcmp(before, r.data, sizeof(before)), 0);
      if (bf16)
        continue;
      for (unsigned high : {0U, 1U})
        for (bool saturate : {false, true})
          for (bool infinity : {false, true}) {
            for (int lane = 0; lane < lanes; ++lane) {
              for (int reg = 0; reg < 8; ++reg) {
                r.v[reg][lane] = infinity ? 0x7c007c00 : 0x7bff7bff;
                r.v[8 + reg][lane] = 0x3c003c00;
              }
              for (int reg = 0; reg < 256 / lanes; ++reg)
                r.v[16 + reg][lane] = 0;
            }
            ASSERT_EQ(fn(exact | (saturate ? GOC_FP16_OVFL : 0), high ? GOC_WMMA_HIGH_C_D : 0,
                         r.v + 24, r.v, r.v + 8, r.v + 16),
                      GOC_SUCCESS);
            for (int reg = 0; reg < 256 / lanes; ++reg)
              for (int lane = 0; lane < lanes; ++lane)
                EXPECT_EQ((r.v[24 + reg][lane] >> (16 * high)) & 65535U,
                          saturate && !infinity ? 0x7bffU : 0x7c00U);
          }
    }
}

TEST(WmmaReplicatedExact, IntegerSignsClampAliasesAndGuards) {
  for (int bits : {4, 8})
    for (int lanes : {32, 64})
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (unsigned mode = 0; mode < 8; ++mode)
          for (int dst : {0, 2, 8, 16, 24})
            for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL}) {
              Wmma fn = lanes == 32 ? (bits == 8 ? goc_v_wmma_i32_16x16x16_iu8
                                                 : goc_v_wmma_i32_16x16x16_iu4)
                                    : (bits == 8 ? goc_v_wmma_i32_16x16x16_iu8_wave64
                                                 : goc_v_wmma_i32_16x16x16_iu4_wave64);
              Registers r;
              int a[16][16], b[16][16];
              for (int row = 0; row < 16; ++row)
                for (int k = 0; k < 16; ++k) {
                  a[row][k] = (row * 7 + k * 13) & ((1 << bits) - 1);
                  b[k][row] = (row * 11 + k * 19) & ((1 << bits) - 1);
                }
              for (int lane = 0; lane < lanes; ++lane) {
                for (int reg = 0; reg < bits / 2; ++reg) {
                  uint32_t av = 0, bv = 0;
                  for (int element = 0; element < 32 / bits; ++element) {
                    int k = reg * (32 / bits) + element;
                    av |= uint32_t(a[(lane % 16 + lane / 16) % 16][k]) << (bits * element);
                    bv |= uint32_t(b[k][(lane % 16 + lane / 16) % 16]) << (bits * element);
                  }
                  r.v[reg][lane] = av;
                  r.v[8 + reg][lane] = bv;
                }
                for (int reg = 0; reg < 256 / lanes; ++reg)
                  r.v[16 + reg][lane] = (lane & 1) ? 0x7ffffff0 : 0x80000010;
              }
              uint32_t expected[32][66];
              std::memcpy(expected, r.data, sizeof(expected));
              for (int reg = 0; reg < 256 / lanes; ++reg)
                for (int lane = 0; lane < lanes; ++lane) {
                  int row = reg * (lanes / 16) + lane / 16, col = lane % 16;
                  int64_t acc = (lane & 1) ? 2147483632LL : -2147483632LL;
                  for (int k = 0; k < 16; ++k) {
                    int x = a[(row + lane / 16) % 16][k], y = b[k][(col + lane / 16) % 16];
                    if ((mode & 1) && x >= (1 << (bits - 1)))
                      x -= 1 << bits;
                    if ((mode & 2) && y >= (1 << (bits - 1)))
                      y -= 1 << bits;
                    acc += x * y;
                    // Captured rocjitsu policy: IU8 pairs, IU4 quartets.
                    if ((mode & 4) && (k + 1) % (bits == 8 ? 2 : 4) == 0)
                      acc = std::clamp(acc, int64_t(-2147483648LL), int64_t(2147483647));
                  }
                  expected[31 - dst - reg][lane + 1] = uint32_t(acc);
                }
              uint64_t modifiers = (mode & 3) | ((mode & 4) ? GOC_WMMA_CLAMP : 0);
              ASSERT_EQ(fn(cpu | semantics | GOC_SEMANTICS_STRICT, modifiers, r.v + dst, r.v,
                           r.v + 8, r.v + 16),
                        GOC_SUCCESS);
              EXPECT_EQ(std::memcmp(r.data, expected, sizeof(expected)), 0);
            }
}

TEST(WmmaReplicatedExact, IntegerClampUsesFormatGroups) {
  for (int bits : {4, 8})
    for (int lanes : {32, 64}) {
      auto fn = lanes == 32
                    ? (bits == 8 ? goc_v_wmma_i32_16x16x16_iu8 : goc_v_wmma_i32_16x16x16_iu4)
                    : (bits == 8 ? goc_v_wmma_i32_16x16x16_iu8_wave64
                                 : goc_v_wmma_i32_16x16x16_iu4_wave64);
      Registers r;
      for (int lane = 0; lane < lanes; ++lane) {
        for (int reg = 0; reg < bits / 2; ++reg) {
          uint32_t av = 0, bv = 0;
          for (int element = 0; element < 32 / bits; ++element) {
            int k = reg * (32 / bits) + element;
            av |= uint32_t(k < 2 ? 1 : k < 4 ? (1 << bits) - 1 : 0) << (bits * element);
            bv |= 1U << (bits * element);
          }
          r.v[reg][lane] = av;
          r.v[8 + reg][lane] = bv;
        }
        for (int reg = 0; reg < 256 / lanes; ++reg)
          r.v[16 + reg][lane] = 0x7fffffff;
      }
      ASSERT_EQ(fn(exact, GOC_WMMA_SIGNED_A | GOC_WMMA_CLAMP, r.v + 24, r.v, r.v + 8, r.v + 16),
                GOC_SUCCESS);
      for (int reg = 0; reg < 256 / lanes; ++reg)
        for (int lane = 0; lane < lanes; ++lane)
          EXPECT_EQ(r.v[24 + reg][lane], bits == 8 ? 0x7ffffffdU : 0x7fffffffU);
      for (unsigned bit = 0; bit < 64; ++bit)
        if (!((1ULL << bit) & (GOC_WMMA_SIGNED_A | GOC_WMMA_SIGNED_B | GOC_WMMA_CLAMP))) {
          EXPECT_EQ(fn(exact, 1ULL << bit, nullptr, nullptr, nullptr, nullptr),
                    GOC_ERROR_INVALID_FLAGS);
        }
    }
}

TEST(WmmaReplicatedExact, IntegerArchitectureSaturationBoundaries) {
  struct Case {
    Wmma fn;
    int bits;
    int lanes;
    bool rdna4;
  };

  const Case cases[] = {
      {goc_v_wmma_i32_16x16x16_iu4, 4, 32, false},
      {goc_v_wmma_i32_16x16x16_iu8, 8, 32, false},
      {goc_v_wmma_i32_16x16x16_iu4_wave64, 4, 64, false},
      {goc_v_wmma_i32_16x16x16_iu8_wave64, 8, 64, false},
      {goc_v_wmma_i32_16x16x16_iu4_rdna4, 4, 32, true},
      {goc_v_wmma_i32_16x16x16_iu8_rdna4, 8, 32, true},
      {goc_v_wmma_i32_16x16x16_iu4_rdna4_wave64, 4, 64, true},
      {goc_v_wmma_i32_16x16x16_iu8_rdna4_wave64, 8, 64, true},
  };
  for (const auto &test : cases)
    for (uint64_t cpu = GOC_CPU_BASELINE; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
        for (bool clamp : {false, true})
          for (bool negative_first : {false, true})
            for (bool signed_b : {false, true})
              for (int dst : {0, 8, 16, 24}) {
                SCOPED_TRACE(::testing::Message()
                             << test.bits << "/" << test.lanes << "/" << test.rdna4 << "/" << cpu
                             << "/" << semantics << "/" << clamp << "/" << negative_first << "/"
                             << signed_b << "/" << dst);
                Registers r;
                // Eight +1 products and eight -1 products cancel. On RDNA4
                // put them in the two physical saturation stages (groups 0+2,
                // then 1+3 in Wave64). RDNA3 uses smaller adjacent groups.
                uint32_t ones = 0;
                for (int shift = 0; shift < 32; shift += test.bits)
                  ones |= 1U << shift;
                const int source_regs =
                    test.rdna4 ? (test.lanes == 64 ? 1 : test.bits / 4) : test.bits / 2;
                for (int reg = 0; reg < source_regs; ++reg)
                  for (int lane = 0; lane < test.lanes; ++lane) {
                    const bool second = test.rdna4 ? (lane / 16) % 2 : reg >= source_regs / 2;
                    const uint32_t factor = (second != negative_first) ? 0xffffffffU : ones;
                    r.v[reg][lane] = signed_b ? ones : factor;
                    r.v[8 + reg][lane] = signed_b ? factor : ones;
                  }
                const uint32_t acc = negative_first ? 0x80000000U : 0x7fffffffU;
                const int output_regs = 256 / test.lanes;
                for (int reg = 0; reg < output_regs; ++reg)
                  std::fill(r.v[16 + reg], r.v[16 + reg] + test.lanes, acc);
                uint32_t expected[32][66];
                std::memcpy(expected, r.data, sizeof(expected));
                const uint32_t result = clamp ? (negative_first ? 0x80000008U : 0x7ffffff7U) : acc;
                for (int reg = 0; reg < output_regs; ++reg)
                  std::fill(expected[31 - dst - reg] + 1, expected[31 - dst - reg] + 1 + test.lanes,
                            result);
                const uint64_t mode = (signed_b ? GOC_WMMA_SIGNED_B : GOC_WMMA_SIGNED_A) |
                                      (clamp ? GOC_WMMA_CLAMP : 0);
                ASSERT_EQ(test.fn(cpu | semantics | GOC_SEMANTICS_STRICT, mode, r.v + dst, r.v,
                                  r.v + 8, r.v + 16),
                          GOC_SUCCESS);
                EXPECT_EQ(std::memcmp(expected, r.data, sizeof(expected)), 0);
              }
}

TEST(WmmaReplicatedExact, FloatingArchitectureRoundingBoundaries) {
  // In the borrowed GFX11 DOT2 model, C+1 rounds back to C before -1
  // arrives in the next step. The GFX12 DOT4 model combines +1 and -1
  // within one step. C is 2^24 for FP32, 2^11 for FP16, 2^8 for BF16.
  for (bool rdna4 : {false, true})
    for (bool bf16 : {false, true})
      for (bool packed : {false, true})
        for (int lanes : {32, 64})
          for (uint64_t cpu = GOC_CPU_BASELINE; cpu <= goc_init_cpu_flags(); ++cpu)
            for (unsigned high = 0; high < (packed && !rdna4 ? 2U : 1U); ++high)
              for (int dst : {0, 4, 8, 16, 24}) {
                SCOPED_TRACE(::testing::Message()
                             << rdna4 << "/" << bf16 << "/" << packed << "/" << lanes << "/" << cpu
                             << "/" << high << "/" << dst);
                Registers r;
                const uint32_t one = bf16 ? 0x3f80 : 0x3c00;
                for (int reg = 0; reg < 8; ++reg)
                  for (int lane = 0; lane < lanes; ++lane) {
                    // K=0 contributes +1, K=2 contributes -1, all others zero.
                    // RDNA3 replicates those elements into every 16-lane group;
                    // RDNA4 holds both in group zero, using different registers.
                    r.v[reg][lane] =
                        (reg > 1 || (rdna4 && lane >= 16)) ? 0 : one | (reg == 1 ? 0x8000U : 0);
                    r.v[8 + reg][lane] = one | (one << 16);
                  }
                const uint32_t acc = packed ? (bf16 ? 0x4380U : 0x6800U) : 0x4b800000U;
                const int output_regs = (packed && rdna4 ? 128 : 256) / lanes;
                const uint32_t c_word = !packed ? acc
                                        : rdna4 ? acc | (acc << 16)
                                                : (acc << (16 * high)) | (0x1234U << (16 * !high));
                for (int reg = 0; reg < output_regs; ++reg)
                  std::fill(r.v[16 + reg], r.v[16 + reg] + lanes, c_word);
                uint32_t expected[32][66];
                std::memcpy(expected, r.data, sizeof(expected));
                const uint32_t value = rdna4    ? acc
                                       : packed ? (bf16 ? 0x437fU : 0x67ffU)
                                                : 0x4b7fffffU;
                for (int reg = 0; reg < output_regs; ++reg)
                  for (int lane = 0; lane < lanes; ++lane) {
                    auto &word = expected[31 - dst - reg][lane + 1];
                    word = !packed ? value
                           : rdna4 ? value | (value << 16)
                                   : (word & ~(0xffffU << (16 * high))) | (value << (16 * high));
                  }
                ASSERT_EQ(floating(bf16, packed, lanes, rdna4)(cpu | exact,
                                                               high ? GOC_WMMA_HIGH_C_D : 0,
                                                               r.v + dst, r.v, r.v + 8, r.v + 16),
                          GOC_SUCCESS);
                EXPECT_EQ(std::memcmp(expected, r.data, sizeof(expected)), 0);
              }
}

TEST(WmmaPackedLoose, LayoutsModifiersAliasesAndGuards) {
  for (bool rdna4 : {false, true})
    for (int lanes : {32, 64})
      for (bool bf16 : {false, true})
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
          for (unsigned mode = 0; mode < (rdna4 ? 64U : 128U); ++mode)
            for (int dst : {0, 8, 16, 24}) {
              Registers r;
              auto encode = [bf16](int value) -> uint16_t {
                return bf16 ? uint16_t(goc::as_bits(float(value)) >> 16)
                            : goc::packed16::pack_f16(value, 0);
              };
              auto av = [](int row, int k, int group) { return (row + 3 * k + group) % 5 - 2; };
              auto bv = [](int k, int col, int group) { return (2 * k + col + 2 * group) % 5 - 2; };
              int registers = (rdna4 ? 128 : 256) / lanes;
              int input_regs = rdna4 ? 128 / lanes : 8;
              for (int lane = 0; lane < lanes; ++lane) {
                for (int reg = 0; reg < input_regs; ++reg) {
                  uint32_t a = 0, b = 0;
                  for (int half = 0; half < 2; ++half) {
                    int k = reg * 2 + half + (rdna4 ? (lane / 16) * (256 / lanes) : 0);
                    int group = rdna4 ? 0 : lane / 16;
                    a |= uint32_t(encode(av(lane % 16, k, group))) << (16 * half);
                    b |= uint32_t(encode(bv(k, lane % 16, group))) << (16 * half);
                  }
                  r.v[reg][lane] = a;
                  r.v[8 + reg][lane] = b;
                }
                for (int reg = 0; reg < registers; ++reg)
                  r.v[16 + reg][lane] = uint32_t(encode(-3)) | (uint32_t(encode(5)) << 16);
              }
              uint32_t expected[32][66];
              std::memcpy(expected, r.data, sizeof(expected));
              for (int row = 0; row < 16; ++row)
                for (int col = 0; col < 16; ++col) {
                  int group = rdna4 ? 0 : row % (lanes / 16);
                  int reg = rdna4 ? (row % (256 / lanes)) / 2 : row / (lanes / 16);
                  int lane = rdna4 ? col + 16 * (row / 8) + (lanes == 64 ? 32 * ((row / 4) % 2) : 0)
                                   : col + 16 * group;
                  int shift = rdna4 ? 16 * (row % 2) : (mode & 64 ? 16 : 0);
                  int sum = shift ? 5 : -3;
                  if (mode & GOC_WMMA_ABS_C)
                    sum = std::abs(sum);
                  if (mode & GOC_WMMA_NEG_C)
                    sum = -sum;
                  for (int k = 0; k < 16; ++k) {
                    int x = av(row, k, group), y = bv(k, col, group);
                    if (mode & (k % 2 ? GOC_WMMA_NEG_HI_A : GOC_WMMA_NEG_LO_A))
                      x = -x;
                    if (mode & (k % 2 ? GOC_WMMA_NEG_HI_B : GOC_WMMA_NEG_LO_B))
                      y = -y;
                    sum += x * y;
                  }
                  auto &word = expected[31 - dst - reg][lane + 1];
                  word = (word & ~(65535U << shift)) | (uint32_t(encode(sum)) << shift);
                }
              uint64_t modifiers = (mode & 63) | ((mode & 64) ? GOC_WMMA_HIGH_C_D : 0);
              ASSERT_EQ(floating(bf16, true, lanes, rdna4)(cpu, modifiers, r.v + dst, r.v, r.v + 8,
                                                           r.v + 16),
                        GOC_SUCCESS);
              EXPECT_EQ(std::memcmp(r.data, expected, sizeof(expected)), 0)
                  << rdna4 << '/' << lanes << '/' << bf16 << '/' << cpu << '/' << mode << '/'
                  << dst;
            }
}

TEST(WmmaPackedLoose, IntermediateNarrowingAndOverflow) {
  for (bool rdna4 : {false, true})
    for (int lanes : {32, 64})
      for (bool bf16 : {false, true})
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
          for (unsigned high = 0; high < (rdna4 ? 1U : 2U); ++high)
            for (int scenario = 0; scenario < (bf16 ? 1 : 5); ++scenario) {
              Registers r;
              // Sixteen 0.5 products: RDNA3 discards each two-product update;
              // RDNA4 retains each four-product update. This distinguishes
              // intermediate narrowing from narrowing just once at the end.
              uint16_t a = bf16 ? 0x3f00 : 0x3800;
              uint16_t b = bf16 ? 0x3f80 : 0x3c00;
              uint16_t c = bf16 ? 0x4380 : 0x6800;
              uint16_t want = rdna4 ? (bf16 ? 0x4384 : 0x6804) : c;
              bool saturate = scenario == 2 || scenario == 4;
              if (scenario) {
                a = scenario >= 3 ? 0x7c00 : 0x7bff;
                c = 0;
                want = saturate && scenario < 3 ? 0x7bff : 0x7c00;
              }
              for (int lane = 0; lane < lanes; ++lane) {
                for (int reg = 0; reg < (rdna4 ? 128 / lanes : 8); ++reg) {
                  r.v[reg][lane] = uint32_t(a) | (uint32_t(a) << 16);
                  r.v[8 + reg][lane] = uint32_t(b) | (uint32_t(b) << 16);
                }
                for (int reg = 0; reg < (rdna4 ? 128 : 256) / lanes; ++reg)
                  r.v[16 + reg][lane] = uint32_t(c) | (uint32_t(c) << 16);
              }
              ASSERT_EQ(floating(bf16, true, lanes, rdna4)(cpu | (saturate ? GOC_FP16_OVFL : 0),
                                                           high ? GOC_WMMA_HIGH_C_D : 0, r.v + 24,
                                                           r.v, r.v + 8, r.v + 16),
                        GOC_SUCCESS);
              for (int reg = 0; reg < (rdna4 ? 128 : 256) / lanes; ++reg)
                for (int lane = 0; lane < lanes; ++lane) {
                  uint32_t word = r.v[24 + reg][lane];
                  EXPECT_EQ((word >> (16 * high)) & 65535U, want);
                  if (rdna4) {
                    EXPECT_EQ(word >> 16, want);
                  }
                }
            }
}

TEST(WmmaPackedLoose, ScalarSimdSpecialAndRandomInputs) {
  const uint16_t half[] = {0,      0x8000, 1,      0x3ff,  0x400,  0x3c00,
                           0xbc00, 0x7bff, 0x7c00, 0xfc00, 0x7c01, 0x7e37};
  const uint16_t bfloat[] = {0,      0x8000, 1,      0x7f,   0x80,   0x3f80,
                             0xbf80, 0x7f7f, 0x7f80, 0xff80, 0x7f81, 0x7fe3};
  for (bool rdna4 : {false, true})
    for (int lanes : {32, 64})
      for (bool bf16 : {false, true})
        for (bool saturate : {false, true})
          for (unsigned batch = 0; batch < 8; ++batch)
            for (uint64_t mode : {0ULL, 1ULL, 32ULL, 63ULL}) {
              Registers original;
              uint32_t rng = 0x12345678U + batch;
              for (int reg = 0; reg < 32; ++reg)
                for (int lane = 0; lane < lanes; ++lane) {
                  uint32_t word = 0;
                  for (int part = 0; part < 2; ++part) {
                    rng = rng * 1664525U + 1013904223U;
                    uint16_t bits = batch < 4
                                        ? (bf16 ? bfloat : half)[(lane + reg + part + batch) % 12]
                                        : uint16_t(rng >> 16);
                    word |= uint32_t(bits) << (16 * part);
                  }
                  original.v[reg][lane] = word;
                }
              const int regs = (rdna4 ? 128 : 256) / lanes;
              const int shift = !rdna4 && (batch & 1) ? 16 : 0;
              const uint64_t modifiers = mode | (shift ? GOC_WMMA_HIGH_C_D : 0);
              const uint64_t flags = saturate ? GOC_FP16_OVFL : 0;
              Registers reference;
              std::memcpy(reference.data, original.data, sizeof(original.data));
              auto fn = floating(bf16, true, lanes, rdna4);
              ASSERT_EQ(fn(flags, modifiers, reference.v + 24, reference.v, reference.v + 8,
                           reference.v + 16),
                        GOC_SUCCESS);
              for (uint64_t cpu = 1; cpu <= goc_init_cpu_flags(); ++cpu) {
                Registers actual;
                std::memcpy(actual.data, original.data, sizeof(original.data));
                ASSERT_EQ(fn(flags | cpu, modifiers, actual.v + 24, actual.v, actual.v + 8,
                             actual.v + 16),
                          GOC_SUCCESS);
                for (int reg = 0; reg < 32; ++reg)
                  for (int word = 0; word < 66; ++word) {
                    uint32_t got = actual.data[31 - reg][word];
                    uint32_t before = original.data[31 - reg][word];
                    if (reg < 24 || reg >= 24 + regs || word == 0 || word > lanes) {
                      EXPECT_EQ(got, before);
                    } else if (!rdna4) {
                      EXPECT_EQ((got >> (16 - shift)) & 65535U, (before >> (16 - shift)) & 65535U);
                    }
                  }
                for (int reg = 0; reg < regs; ++reg)
                  for (int lane = 0; lane < lanes; ++lane)
                    for (int part = 0; part < (rdna4 ? 2 : 1); ++part) {
                      uint16_t want = reference.v[24 + reg][lane] >> (shift + 16 * part);
                      uint16_t got = actual.v[24 + reg][lane] >> (shift + 16 * part);
                      unsigned infinity = bf16 ? 0x7f80 : 0x7c00;
                      if ((want & 0x7fffU) > infinity) {
                        EXPECT_GT(got & 0x7fffU, infinity);
                      } else if ((want & 0x7fffU) != 0 && (want & 0x7fffU) < infinity) {
                        // Independent arithmetic decoding for the comparison. The
                        // expected arithmetic is still a scalar/SIMD differential.
                        auto decode = [bf16](uint16_t bits) {
                          unsigned mantissa = bf16 ? 7 : 10, bias = bf16 ? 127 : 15;
                          unsigned exponent = (bits & 32767) >> mantissa;
                          unsigned fraction = bits & ((1U << mantissa) - 1);
                          if (exponent == (bf16 ? 255U : 31U))
                            return fraction ? NAN : (bits & 32768 ? -INFINITY : INFINITY);
                          double value =
                              std::ldexp(double(fraction + (exponent ? 1U << mantissa : 0)),
                                         int(exponent ? exponent : 1) - int(bias) - int(mantissa));
                          return float(bits & 32768 ? -value : value);
                        };
                        EXPECT_TRUE(
                            goc_test::fuzzy(decode(got), decode(want),
                                            bf16 ? goc_test::half_dot_bf16_tolerance
                                                 : goc_test::scalar_arithmetic_f16_tolerance))
                            << rdna4 << '/' << lanes << '/' << bf16 << '/' << batch << '/' << mode
                            << '/' << cpu;
                      } else {
                        EXPECT_EQ(got, want) << rdna4 << '/' << lanes << '/' << bf16 << '/' << batch
                                             << '/' << mode << '/' << cpu;
                      }
                    }
              }
            }
}

// Reconstruct rocjitsu rdna_wmma_i32_test.cpp's physical capture inputs.
// Expected words are hardware observations, not results of a GoC oracle.
TEST(WmmaReplicatedExact, CapturedIntegerSaturation) {
  struct Case {
    Wmma fn;
    int lanes;
    bool rdna4;
    const std::array<uint32_t, 256> *expected;
  };

  const Case cases[] = {
      {goc_v_wmma_i32_16x16x16_iu8, 32, false, &kRdnaWmmaIu8Saturated[0]},
      {goc_v_wmma_i32_16x16x16_iu8_wave64, 64, false, &kRdnaWmmaIu8Saturated[1]},
      {goc_v_wmma_i32_16x16x16_iu8_rdna4, 32, true, &kRdnaWmmaIu8Saturated[2]},
      {goc_v_wmma_i32_16x16x16_iu8_rdna4_wave64, 64, true, &kRdnaWmmaIu8Saturated[3]},
      {goc_v_wmma_i32_16x16x16_iu4, 32, false, &kRdnaWmmaIu4Saturated[0]},
      {goc_v_wmma_i32_16x16x16_iu4_wave64, 64, false, &kRdnaWmmaIu4Saturated[1]},
      {goc_v_wmma_i32_16x16x16_iu4_rdna4, 32, true, &kRdnaWmmaIu4Saturated[2]},
      {goc_v_wmma_i32_16x16x16_iu4_rdna4_wave64, 64, true, &kRdnaWmmaIu4Saturated[3]},
      {goc_v_wmma_i32_16x16x32_iu4, 32, true, &kRdnaWmmaIu4K32Saturated[0]},
      {goc_v_wmma_i32_16x16x32_iu4_wave64, 64, true, &kRdnaWmmaIu4K32Saturated[1]},
  };
  for (const auto &test : cases)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
        for (int dst : {0, 8, 16, 24}) {
          SCOPED_TRACE(::testing::Message() << test.lanes << "/" << test.rdna4 << "/" << cpu << "/"
                                            << semantics << "/" << dst);
          Registers r;
          uint32_t rng = 0x12056101U;
          const uint32_t accumulators[] = {0,           1,           0xffffffffU, 0x7fffffffU,
                                           0x80000000U, 0x7ffff000U, 0x80001000U, 0x12345678U};
          for (int reg = 0; reg < 32; ++reg)
            for (int lane = 0; lane < test.lanes; ++lane) {
              rng ^= rng << 13;
              rng ^= rng >> 17;
              rng ^= rng << 5;
              r.v[reg][lane] = reg >= 16 ? accumulators[(rng >> 24) & 7] : rng;
            }
          if (!test.rdna4)
            for (int reg = 0; reg < 16; ++reg)
              for (int lane = 16; lane < test.lanes; ++lane)
                r.v[reg][lane] = r.v[reg][lane % 16];
          uint32_t expected[32][66];
          std::memcpy(expected, r.data, sizeof(expected));
          for (int reg = 0; reg < 256 / test.lanes; ++reg)
            for (int lane = 0; lane < test.lanes; ++lane)
              expected[31 - dst - reg][lane + 1] = (*test.expected)[reg * test.lanes + lane];
          ASSERT_EQ(test.fn(cpu | semantics | GOC_SEMANTICS_STRICT,
                            GOC_WMMA_SIGNED_A | GOC_WMMA_SIGNED_B | GOC_WMMA_CLAMP, r.v + dst, r.v,
                            r.v + 8, r.v + 16),
                    GOC_SUCCESS);
          EXPECT_EQ(std::memcmp(expected, r.data, sizeof(expected)), 0);
        }
}

TEST(WmmaReplicatedExact, CapturedDenseMatricesAndOperands) {
  goc_test::ScopedFpEnvironment restore;
  ASSERT_TRUE(restore.saved());
  for (unsigned host_case = 0; host_case < 8; ++host_case) {
    ASSERT_NO_FATAL_FAILURE(goc_test::matrix_test_host(host_case));
    const unsigned controls = goc_test::matrix_host_controls();

    struct Shape {
      bool bf16, packed;

      const RdnaWmmaFloatMatrixCase *matrices;
      unsigned matrix_count;
      const RdnaWmmaFloatOperandCase *operands;
      unsigned operand_count;
    };

    const Shape shapes[] = {
        {false, false, kRdnaWmmaF32F16Matrices.data(), kRdnaWmmaF32F16Matrices.size(),
         kRdnaWmmaF32F16Operands.data(), kRdnaWmmaF32F16Operands.size()},
        {true, false, kRdnaWmmaF32Bf16Matrices.data(), kRdnaWmmaF32Bf16Matrices.size(),
         kRdnaWmmaF32Bf16Operands.data(), kRdnaWmmaF32Bf16Operands.size()},
        {false, true, kRdnaWmmaF16F16Matrices.data(), kRdnaWmmaF16F16Matrices.size(),
         kRdnaWmmaF16F16Operands.data(), kRdnaWmmaF16F16Operands.size()},
        {true, true, kRdnaWmmaBf16Bf16Matrices.data(), kRdnaWmmaBf16Bf16Matrices.size(),
         kRdnaWmmaBf16Bf16Operands.data(), kRdnaWmmaBf16Bf16Operands.size()},
    };
    unsigned replayed = 0;
    for (const auto &shape : shapes)
      for (unsigned sample = 0; sample < shape.matrix_count + shape.operand_count; ++sample)
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
          const bool matrix = sample < shape.matrix_count;
          const auto &m = shape.matrices[matrix ? sample : 0];
          const auto &o = shape.operands[matrix ? 0 : sample - shape.matrix_count];
          const unsigned width = matrix ? m.width : o.width;
          const auto code = matrix ? m.code : o.code;
          const bool rdna4 = matrix ? m.gfx12 : o.gfx12;
          if (cpu == 0)
            ++replayed;
          SCOPED_TRACE(::testing::Message() << shape.bf16 << shape.packed << "/" << sample << "/"
                                            << width << "/" << cpu);
          uint32_t data[33][66], expected[33][66];
          uint32_t *v[33];
          uint32_t rng = m.seed;
          constexpr uint32_t accumulators[] = {0,           1,           0x80000000U, 0x3f800001U,
                                               0xbf800001U, 0x43800001U, 0xc3800001U, 0x80000001U};
          constexpr uint32_t pairs[] = {4, 8, 12, 9, 13, 14};
          for (unsigned reg = 0; reg < 33; ++reg) {
            v[reg] = data[reg] + 1;
            std::fill(data[reg], data[reg] + 66, 0xdeadbeefU);
            for (unsigned lane = 0; lane < width; ++lane) {
              uint32_t value;
              if (matrix) {
                rng ^= rng << 13;
                rng ^= rng >> 17;
                rng ^= rng << 5;
                value = rng;
                if (m.finite && reg < 16) {
                  const unsigned max_finite = shape.bf16 ? 0x7f7f : 0x7bff;
                  value = 0;
                  for (unsigned shift = 0; shift < 32; shift += 16) {
                    unsigned part = (rng >> shift) & 65535;
                    value |= ((part & 32768) | (m.bounded_bf16 ? 0x3c80U + (part & 32767) % 0x500U
                                                               : (part & 32767) % (max_finite + 1)))
                             << shift;
                  }
                } else if (m.finite && reg < 32) {
                  value = accumulators[(rng >> 24) & 7];
                  if (m.bounded_bf16 && shape.packed) {
                    constexpr uint32_t packed_c[] = {0,           0x00010001U, 0x80008000U,
                                                     0x3f813f81U, 0xbf81bf81U, 0x43814381U,
                                                     0xc381c381U, 0x80018001U};
                    value = packed_c[(rng >> 24) & 7];
                  }
                }
                if (reg == 32) {
                  value = 0;
                  for (unsigned i = 0; i < 8; ++i)
                    value |= pairs[((rng >> (4 * i)) & 15) % 6] << (4 * i);
                }
              } else
                value = reg < 8 ? o.a : reg < 16 ? o.b : reg < 24 ? o.c : reg < 32 ? o.d : o.index;
              v[reg][lane] = value;
            }
          }
          if (!rdna4)
            for (unsigned reg = 0; reg < 16; ++reg)
              for (unsigned lane = 16; lane < width; ++lane)
                v[reg][lane] = v[reg][lane % 16];
          if (!matrix && o.factor_base != RdnaWmmaFloatOperandCase::kUniformFactors)
            for (unsigned i = 0; i < 128; ++i) {
              unsigned offset = o.factor_base * width + i;
              v[offset / width][offset % width] = o.factor_groups[i / 16];
            }
          std::memcpy(expected, data, sizeof(expected));
          unsigned dest = code[0] & 255;
          unsigned count = matrix ? m.expected_words : shape.packed && rdna4 ? 128 : 256;
          for (unsigned i = 0; i < count; ++i)
            expected[dest + i / width][1 + i % width] = matrix ? m.expected[i] : o.expected;
          unsigned a = (code[1] & 511) - 256, b = ((code[1] >> 9) & 511) - 256,
                   c = (code[1] >> 18) & 511;
          ASSERT_LT(a, 33U);
          ASSERT_LT(b, 33U);
          uint32_t inline_c[8][64];
          const uint32_t *cv[8];
          if (c >= 256) {
            ASSERT_LT(c - 256, 33U);
            for (unsigned i = 0; i < (shape.packed && rdna4 ? 128U : 256U) / width; ++i)
              cv[i] = v[c - 256 + i];
          } else {
            ASSERT_TRUE(c == 128 || c == 193 || c == 240);
            for (unsigned i = 0; i < 8; ++i) {
              uint32_t word = c == 128 ? 0 : c == 193 ? 0xffffffffU : 0x3f000000U;
              // Floating inline constants use repeated packed format bits.
              if (shape.packed && c == 240)
                word = shape.bf16 ? 0x3f003f00U : 0x38003800U;
              std::fill(inline_c[i], inline_c[i] + 64, word);
              cv[i] = inline_c[i];
            }
          }
          uint64_t modifiers = ((code[1] >> 29) & 7) | (((code[0] >> 8) & 7) << 3);
          if (!rdna4 && shape.packed && (code[0] & (1U << 13)))
            modifiers |= GOC_WMMA_HIGH_C_D;
          auto fn = floating(shape.bf16, shape.packed, width, rdna4);
          ASSERT_EQ(fn(cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT |
                           goc_test::matrix_capture_flags(matrix ? m.mode : o.mode, host_case),
                       modifiers, v + dest, v + a, v + b, cv),
                    GOC_SUCCESS);
          for (unsigned reg = 0; reg < 33; ++reg)
            for (unsigned lane = 0; lane < 66; ++lane)
              ASSERT_EQ(data[reg][lane], expected[reg][lane])
                  << "reg=" << reg << " guarded lane=" << lane;
        }
    EXPECT_EQ(replayed, 1618U);
    EXPECT_EQ(goc_test::matrix_host_controls(), controls);
  }
}

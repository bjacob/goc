// SPDX-License-Identifier: MIT

#include "fp_environment.h"
#include "gfx11_dot2.h"
#include "gfx11_dot2_fixtures.h"
#include "goc/goc.h"
#include "internal.h"
#include "packed16.h"
#include "packed_wmma_fixtures.h"
#include "wmma_replicated_packed_fixtures.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>

namespace {

using Wmma = decltype(&goc_v_wmma_f32_16x16x16_f16);

Wmma floating(bool bf16, bool packed, int lanes) {
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
      for (int dst : {0, 8, 16, 24}) {
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
        ASSERT_EQ(floating(bf16, false, lanes)(exact, 0, r.v + dst, r.v, r.v + 8, r.v + 16),
                  GOC_SUCCESS);
        for (int reg = 0; reg < 256 / lanes; ++reg)
          for (int lane = 0; lane < lanes; ++lane)
            EXPECT_EQ(r.v[dst + reg][lane],
                      kWmmaIntegerResult[(reg * (lanes / 16) + lane / 16) * 16 + lane % 16]);
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
              ASSERT_EQ(floating(bf16, true, lanes)(exact, high ? GOC_WMMA_HIGH_C_D : 0, r.v + dst,
                                                    r.v, r.v + 8, r.v + 16),
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
      for (unsigned bit = 0; bit < 64; ++bit) {
        if ((1ULL << bit) & (63ULL | GOC_WMMA_HIGH_C_D))
          continue;
        EXPECT_EQ(fn(exact, 1ULL << bit, nullptr, nullptr, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
      }
      EXPECT_EQ(
          fn(GOC_SEMANTICS_MASK | GOC_SEMANTICS_STRICT, 0, nullptr, nullptr, nullptr, nullptr),
          GOC_ERROR_UNSUPPORTED_SEMANTICS);
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
      for (unsigned mode = 0; mode < 8; ++mode)
        for (int dst : {0, 2, 8, 16, 24})
          for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL}) {
            Wmma fn = lanes == 32
                          ? (bits == 8 ? goc_v_wmma_i32_16x16x16_iu8 : goc_v_wmma_i32_16x16x16_iu4)
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
                  av |= uint32_t(a[lane % 16][k]) << (bits * element);
                  bv |= uint32_t(b[k][lane % 16]) << (bits * element);
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
                  int x = a[row][k], y = b[k][col];
                  if ((mode & 1) && x >= (1 << (bits - 1)))
                    x -= 1 << bits;
                  if ((mode & 2) && y >= (1 << (bits - 1)))
                    y -= 1 << bits;
                  acc += x * y;
                }
                if (mode & 4)
                  acc = std::clamp(acc, int64_t(-2147483648LL), int64_t(2147483647));
                expected[31 - dst - reg][lane + 1] = uint32_t(acc);
              }
            uint64_t modifiers = (mode & 3) | ((mode & 4) ? GOC_WMMA_CLAMP : 0);
            ASSERT_EQ(
                fn(semantics | GOC_SEMANTICS_STRICT, modifiers, r.v + dst, r.v, r.v + 8, r.v + 16),
                GOC_SUCCESS);
            EXPECT_EQ(std::memcmp(r.data, expected, sizeof(expected)), 0);
          }
}

TEST(WmmaReplicatedExact, IntegerClampIsFinalNotStaged) {
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
            av |= uint32_t(k < 8 ? 1 : (1 << bits) - 1) << (bits * element);
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
          EXPECT_EQ(r.v[24 + reg][lane], 0x7fffffffU);
      for (unsigned bit = 0; bit < 64; ++bit)
        if (!((1ULL << bit) & (GOC_WMMA_SIGNED_A | GOC_WMMA_SIGNED_B | GOC_WMMA_CLAMP))) {
          EXPECT_EQ(fn(exact, 1ULL << bit, nullptr, nullptr, nullptr, nullptr),
                    GOC_ERROR_INVALID_FLAGS);
        }
    }
}

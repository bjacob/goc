// SPDX-License-Identifier: MIT

#include "goc.h"
#include "rdna4_packed_modifier_fixtures.h"
#include "rdna4_packed_wmma_fixtures.h"

#include <algorithm>
#include <cfenv>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdint.h>

namespace {

using Wmma = decltype(&goc_rdna4_v_wmma_f16_16x16x16_f16);

Wmma function(int width, bool bf16) {
  if (width == 64)
    return bf16 ? goc_rdna4w64_v_wmma_bf16_16x16x16_bf16 : goc_rdna4w64_v_wmma_f16_16x16x16_f16;
  return bf16 ? goc_rdna4_v_wmma_bf16_16x16x16_bf16 : goc_rdna4_v_wmma_f16_16x16x16_f16;
}

struct Registers {
  uint32_t storage[16][67] = {};
  uint32_t *v[16];

  Registers() {
    for (int i = 0; i < 16; ++i) {
      std::fill(storage[i], storage[i] + 67, 0xdeadbeef);
      v[i] = storage[15 - i] + 1;
    }
  }

  void guards(int width) {
    for (auto &r : storage) {
      EXPECT_EQ(r[0], 0xdeadbeef);
      for (int lane = width + 1; lane < 67; ++lane)
        EXPECT_EQ(r[lane], 0xdeadbeef);
    }
  }
};

// Logical matrices are stored independently of the implementation's readers.
void load(Registers &r, const PackedWmmaInput &input, int width) {
  for (int lane = 0; lane < width; ++lane) {
    for (int reg = 0; reg < 128 / width; ++reg) {
      int k = (lane / 16) * (256 / width) + 2 * reg;
      int index = lane % 16;
      r.v[reg][lane] = input.a[index * 16 + k] | (uint32_t(input.a[index * 16 + k + 1]) << 16);
      r.v[4 + reg][lane] =
          input.b[k * 16 + index] | (uint32_t(input.b[(k + 1) * 16 + index]) << 16);
      int row = width == 32 ? (lane / 16) * 8 + 2 * reg
                            : ((lane >> 4) & 1) * 8 + (lane >> 5) * 4 + 2 * reg;
      r.v[8 + reg][lane] =
          input.c[row * 16 + index] | (uint32_t(input.c[(row + 1) * 16 + index]) << 16);
    }
  }
}

struct HostState {
  std::fenv_t saved;

  HostState() { std::fegetenv(&saved); }

  ~HostState() { std::fesetenv(&saved); }
};

} // namespace

TEST(PackedWmma, HardwareMatricesMasksOverlapAndHostState) {
  HostState restore;
  for (int width : {32, 64})
    for (bool bf16 : {false, true})
      for (int fixture = 0; fixture < 7; ++fixture)
        for (int dst : {0, 4, 8, 12})
          for (uint64_t mask : {UINT64_C(0), UINT64_C(0xa55a0123fedc9876), UINT64_MAX})
            for (int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
              Registers r;
              load(r, kPackedInputs[bf16][fixture], width);
              uint32_t before[4][64];
              for (int reg = 0; reg < 128 / width; ++reg)
                std::copy(r.v[dst + reg], r.v[dst + reg] + width, before[reg]);
              std::fesetround(rounding);
              std::feclearexcept(FE_ALL_EXCEPT);
              std::feraiseexcept(FE_INEXACT);
              ASSERT_EQ(function(width, bf16)(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                                              mask, 0, r.v + dst, r.v, r.v + 4, r.v + 8),
                        GOC_SUCCESS);
              EXPECT_EQ(std::fegetround(), rounding);
              EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_INEXACT);
              for (int row = 0; row < 16; ++row)
                for (int col = 0; col < 16; ++col) {
                  int lane = col + 16 * (row / 8) + (width == 64 ? 32 * ((row / 4) % 2) : 0);
                  int reg = (row % (256 / width)) / 2, shift = 16 * (row % 2);
                  uint32_t want = (mask >> lane) & 1
                                      ? kPackedExpected[width == 64][bf16][fixture][row * 16 + col]
                                      : ((before[reg][lane] >> shift) & 65535);
                  EXPECT_EQ((r.v[dst + reg][lane] >> shift) & 65535, want)
                      << "width=" << width << " bf16=" << bf16 << " fixture=" << fixture
                      << " row=" << row << " col=" << col;
                }
              r.guards(width);
            }
}

TEST(PackedWmma, OverflowStateAndModifiers) {
  // Finite 256*256 overflows FP16. The GPU state selects infinity or 65504.
  // A true input infinity remains infinity in either mode.
  for (int width : {32, 64})
    for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
      for (bool saturate : {false, true})
        for (bool infinite : {false, true})
          for (bool negative : {false, true}) {
            Registers r;
            PackedWmmaInput input{};
            for (int i = 0; i < 16; ++i) {
              input.a[i * 16] = infinite ? 0x7c00 : 0x5c00;
              input.b[i] = 0x5c00;
            }
            load(r, input, width);
            auto flags = semantics | GOC_SEMANTICS_STRICT | (saturate ? GOC_FP16_OVFL : 0);
            ASSERT_EQ(function(width, false)(flags, UINT64_MAX, negative ? GOC_WMMA_NEG_LO_A : 0,
                                             r.v + 12, r.v, r.v + 4, r.v + 8),
                      0);
            uint32_t want = (saturate && !infinite ? 0x7bff : 0x7c00) | (negative ? 0x8000 : 0);
            for (int reg = 0; reg < 128 / width; ++reg)
              for (int lane = 0; lane < width; ++lane)
                EXPECT_EQ(r.v[12 + reg][lane], want | (want << 16));
          }
}

TEST(PackedWmma, EveryModifierCombinationAndValidation) {
  for (int width : {32, 64})
    for (bool bf16 : {false, true})
      for (uint32_t modifiers = 0; modifiers < 64; ++modifiers) {
        Registers r;
        PackedWmmaInput input{};
        const uint16_t one = bf16 ? 0x3f80 : 0x3c00;
        for (int i = 0; i < 256; ++i) {
          input.a[i] = input.b[i] = one;
          input.c[i] = one | 0x8000;
        }
        load(r, input, width);
        ASSERT_EQ(function(width, bf16)(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                                        UINT64_MAX, modifiers, r.v + 12, r.v, r.v + 4, r.v + 8),
                  0);
        int sum = 8 * ((bool(modifiers & 1) != bool(modifiers & 2)) ? -1 : 1) +
                  8 * ((bool(modifiers & 8) != bool(modifiers & 16)) ? -1 : 1);
        int c = (modifiers & GOC_WMMA_ABS_C) ? 1 : -1;
        sum += (modifiers & GOC_WMMA_NEG_C) ? -c : c;
        // All results are +/-1, +/-15 or +/-17, exactly representable.
        int mag = sum < 0 ? -sum : sum;
        uint32_t want = bf16 ? (mag == 1    ? 0x3f80
                                : mag == 15 ? 0x4170
                                            : 0x4188)
                             : (mag == 1    ? 0x3c00
                                : mag == 15 ? 0x4b80
                                            : 0x4c40);
        if (sum < 0)
          want |= 0x8000;
        for (int reg = 0; reg < 128 / width; ++reg)
          for (int lane = 0; lane < width; ++lane)
            EXPECT_EQ(r.v[12 + reg][lane], want | (want << 16));
        uint32_t old = r.v[12][0];
        EXPECT_EQ(function(width, bf16)(0, UINT64_MAX, 64, r.v + 12, r.v, r.v + 4, r.v + 8),
                  GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(function(width, bf16)((UINT64_C(2) << 16) | GOC_SEMANTICS_STRICT, UINT64_MAX, 0,
                                        r.v + 12, r.v, r.v + 4, r.v + 8),
                  GOC_ERROR_UNSUPPORTED_SEMANTICS);
        EXPECT_EQ(r.v[12][0], old);
      }
}

TEST(PackedWmma, HardwareModifierCaptures) {
  for (int width : {32, 64})
    for (const auto &f : kPackedModifiers) {
      Registers r;
      for (int reg = 0; reg < 128 / width; ++reg)
        for (int lane = 0; lane < width; ++lane) {
          r.v[reg][lane] = f.a;
          r.v[4 + reg][lane] = f.b;
          r.v[8 + reg][lane] = f.c;
        }
      ASSERT_EQ(function(width, f.bf16)(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                                        UINT64_MAX, f.modifiers, r.v + 12, r.v, r.v + 4, r.v + 8),
                0);
      for (int reg = 0; reg < 128 / width; ++reg)
        for (int lane = 0; lane < width; ++lane)
          EXPECT_EQ(r.v[12 + reg][lane], f.expected) << "modifiers=" << f.modifiers;
    }
}

TEST(PackedWmma, HardwareIntermediateOverflowState) {
  // Adapted from rocjitsu PackedWmma.HardwareOverflowModeAtIntermediateSteps.
  // Two maximum*2 products overflow the first dot4. A later opposite-sign
  // maximum cancels saturated FP16 to zero, but cannot cancel infinity.
  HostState restore;
  for (int width : {32, 64})
    for (bool bf16 : {false, true})
      for (bool saturate : {false, true})
        for (int pattern = 0; pattern < 4; ++pattern)
          for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
            Registers r;
            PackedWmmaInput input{};
            uint16_t maximum = bf16 ? 0x7f7f : 0x7bff;
            uint16_t infinity = bf16 ? 0x7f80 : 0x7c00;
            for (int index = 0; index < 16; ++index) {
              for (int k = 0; k < 2; ++k) {
                input.a[index * 16 + k] =
                    (pattern == 3 ? infinity : maximum) | (pattern == 2 ? 0x8000 : 0);
                input.b[k * 16 + index] = 0x4000;
              }
              if (pattern) {
                // The second dot4 begins at physical K=8 in wave32 and K=4 in wave64.
                int k = width == 32 ? 8 : 4;
                input.a[index * 16 + k] = maximum | (pattern == 2 ? 0 : 0x8000);
                input.b[k * 16 + index] = bf16 ? 0x3f80 : 0x3c00;
              }
            }
            load(r, input, width);
            std::fesetround(rounding);
            std::feclearexcept(FE_ALL_EXCEPT);
            std::feraiseexcept(FE_INEXACT);
            uint64_t flags = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT |
                             (saturate ? GOC_FP16_OVFL : 0);
            ASSERT_EQ(function(width, bf16)(flags, UINT64_MAX, 0, r.v + 12, r.v, r.v + 4, r.v + 8),
                      0);
            EXPECT_EQ(std::fegetround(), rounding);
            EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_INEXACT);
            uint32_t expected = infinity | (pattern == 2 ? 0x8000 : 0);
            if (!bf16 && saturate && pattern != 3)
              expected = pattern == 0 ? 0x7bff : 0;
            for (int reg = 0; reg < 128 / width; ++reg)
              for (int lane = 0; lane < width; ++lane)
                EXPECT_EQ(r.v[12 + reg][lane], expected | (expected << 16))
                    << "width=" << width << " bf16=" << bf16 << " saturate=" << saturate
                    << " pattern=" << pattern;
          }
}

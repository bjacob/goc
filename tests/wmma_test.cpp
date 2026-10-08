// SPDX-License-Identifier: MIT
#include "dot_fixtures.h"
#include "float_formats.h"
#include "rdna4_dot.h"
#include "wmma_fixtures.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
namespace {
using Wmma = decltype(&goc_rdna4_v_wmma_f32_16x16x16_f16);
// Each logical VGPR is deliberately separated by padding and allocated in
// reverse order, exercising the API's array-of-pointers contract.
struct Registers {
  uint32_t storage[24][35] = {};
  uint32_t *v[24];
  Registers() {
    for (int i = 0; i < 24; ++i) {
      v[i] = storage[23 - i] + 1;
      storage[i][0] = storage[i][34] = 0xdeadbeef;
    }
  }
  void guards() {
    for (auto &r : storage) {
      EXPECT_EQ(r[0], 0xdeadbeef);
      EXPECT_EQ(r[34], 0xdeadbeef);
    }
  }
};
void set16(uint32_t *const *v, int index, int k, uint16_t bits) {
  auto &word = v[k % 8 / 2][index + 16 * (k / 8)];
  int shift = 16 * (k % 2);
  word = (word & ~(0xffffu << shift)) | (uint32_t(bits) << shift);
}
bool fuzzy(float actual, float expected) {
  // NumPy-style isclose, as used by hrx-system's math.h.
  return actual == expected || (std::isnan(actual) && std::isnan(expected)) ||
         (std::isfinite(expected) &&
          std::abs(double(actual) - expected) <= 1e-5 + 1e-5 * std::abs(double(expected)));
}
} // namespace
TEST(Wmma, HardwareCapturedLooseResults) {
  for (bool bf16 : {false, true}) {
    auto fn = bf16 ? goc_rdna4_v_wmma_f32_16x16x16_bf16 : goc_rdna4_v_wmma_f32_16x16x16_f16;
    const auto &fixtures = bf16 ? kGfx12WmmaBF16Cases : kGfx12WmmaF16Cases;
    for (const auto &f : fixtures)
      for (int dst : {0, 4, 8, 16}) {
        Registers r;
        for (int i = 0; i < 16; ++i)
          for (int k = 0; k < 16; ++k) {
            set16(r.v, i, k, f.a[k]);
            set16(r.v + 4, i, k, f.b[k]);
          }
        for (int reg = 0; reg < 8; ++reg)
          std::fill(r.v[8 + reg], r.v[8 + reg] + 32, f.c);
        ASSERT_EQ(fn(0, UINT32_MAX, 0, r.v + dst, r.v, r.v + 4, r.v + 8), 0);
        for (int reg = 0; reg < 8; ++reg)
          for (int lane = 0; lane < 32; ++lane)
            EXPECT_TRUE(fuzzy(goc::as_float(r.v[dst + reg][lane]), goc::as_float(f.expected32)));
        r.guards();
      }
  }
}
TEST(Wmma, LaneMappingMaskAndPartialOperandOverlap) {
  for (bool bf16 : {false, true})
    for (uint32_t mask : {0u, 1u, 0xaaaaaaaa, 0xffffffff}) {
      Registers r;
      const uint16_t one = bf16 ? 0x3f80 : 0x3c00;
      // A is identity. B encodes a row/column-dependent power of two so that
      // every D coordinate has an independently known, exactly representable value.
      for (int row = 0; row < 16; ++row)
        for (int k = 0; k < 16; ++k)
          set16(r.v, row, k, row == k ? one : 0);
      for (int col = 0; col < 16; ++col)
        for (int k = 0; k < 16; ++k)
          set16(r.v + 4, col, k, uint16_t(one + ((k + col) % 8) * (bf16 ? 128 : 1024)));
      // D starts at register 2, so it shares only some VGPRs with A, B and C.
      std::array<std::array<uint32_t, 32>, 8> old;
      for (int reg = 0; reg < 8; ++reg)
        std::copy(r.v[2 + reg], r.v[2 + reg] + 32, old[reg].begin());
      auto fn = bf16 ? goc_rdna4_v_wmma_f32_16x16x16_bf16 : goc_rdna4_v_wmma_f32_16x16x16_f16;
      ASSERT_EQ(fn(0, mask, 0, r.v + 2, r.v, r.v + 4, r.v + 8), 0);
      for (int lane = 0; lane < 32; ++lane)
        for (int reg = 0; reg < 8; ++reg) {
          int row = reg + 8 * (lane / 16), col = lane % 16;
          uint32_t expected = ((mask >> lane) & 1) ? goc::as_bits(std::ldexp(1.0f, (row + col) % 8))
                                                   : old[reg][lane];
          EXPECT_EQ(r.v[2 + reg][lane], expected) << "lane=" << lane << " reg=" << reg;
        }
      r.guards();
    }
}
TEST(Wmma, UnsupportedSemanticsPreserveAllRegisters) {
  Registers r;
  for (auto &reg : r.storage)
    for (auto &x : reg)
      x = 0xdeadbeef;
  EXPECT_EQ(goc_rdna4_v_wmma_f32_16x16x16_f16(GOC_SEMANTICS_MASK | GOC_SEMANTICS_STRICT, UINT32_MAX,
                                              0, r.v, r.v, r.v, r.v),
            GOC_ERROR_UNSUPPORTED_SEMANTICS);
  for (auto &reg : r.storage)
    for (auto x : reg)
      EXPECT_EQ(x, 0xdeadbeef);
}
TEST(FloatFormats, HalfBoundaryBits) {
  const uint16_t in[] = {0, 0x8000, 1, 0x03ff, 0x0400, 0x3c00, 0x7bff, 0x7c00, 0xfc00, 0x7e00};
  const uint32_t out[] = {0,          0x80000000, 0x33800000, 0x387fc000, 0x38800000,
                          0x3f800000, 0x477fe000, 0x7f800000, 0xff800000, 0x7fc00000};
  for (int i = 0; i < 10; ++i)
    EXPECT_EQ(goc::as_bits(goc::f16_to_float(in[i])), out[i]);
}

TEST(Wmma, HardwareCapturedExactResults) {
  for (bool bf16 : {false, true}) {
    auto fn = bf16 ? goc_rdna4_v_wmma_f32_16x16x16_bf16 : goc_rdna4_v_wmma_f32_16x16x16_f16;
    const auto &fixtures = bf16 ? kGfx12WmmaBF16Cases : kGfx12WmmaF16Cases;
    for (const auto &f : fixtures)
      for (int dst : {0, 4, 8, 16})
        for (uint32_t mask : {0u, 0x55555555u, 0xffffffffu}) {
          Registers r;
          for (int i = 0; i < 16; ++i)
            for (int k = 0; k < 16; ++k) {
              set16(r.v, i, k, f.a[k]);
              set16(r.v + 4, i, k, f.b[k]);
            }
          for (int reg = 0; reg < 8; ++reg)
            std::fill(r.v[8 + reg], r.v[8 + reg] + 32, f.c);
          std::array<std::array<uint32_t, 32>, 8> old;
          for (int reg = 0; reg < 8; ++reg)
            std::copy(r.v[dst + reg], r.v[dst + reg] + 32, old[reg].begin());
          ASSERT_EQ(fn(GOC_SEMANTICS_EXACT | GOC_SEMANTICS_STRICT, mask, 0, r.v + dst, r.v, r.v + 4,
                       r.v + 8),
                    0);
          for (int reg = 0; reg < 8; ++reg)
            for (int lane = 0; lane < 32; ++lane)
              EXPECT_EQ(r.v[dst + reg][lane], ((mask >> lane) & 1) ? f.expected32 : old[reg][lane]);
          r.guards();
        }
  }
}
TEST(Rdna4Dot, HardwareCapturedSpecialValuesAndRounding) {
  for (const auto &f : kGfx12DotF16Cases) {
    std::array<uint16_t, 2> a = {uint16_t(f.a), uint16_t(f.a >> 16)},
                            b = {uint16_t(f.b), uint16_t(f.b >> 16)};
    EXPECT_EQ((goc::gfx12_dot_bits<false, 2>(a, b, f.c)), f.expected);
  }
  for (const auto &f : kGfx12DotBF16Cases) {
    std::array<uint16_t, 2> a = {uint16_t(f.a), uint16_t(f.a >> 16)},
                            b = {uint16_t(f.b), uint16_t(f.b >> 16)};
    EXPECT_EQ((goc::gfx12_dot_bits<true, 2>(a, b, f.c)), f.expected);
  }
}

TEST(Rdna4Dot, PublicApiHardwareFixturesMaskAndOverlap) {
  const auto check = [](const auto &cases, Wmma fn) {
    for (const auto &f : cases)
      for (int dst : {0, 1, 2, 3}) {
        Registers r;
        for (int lane = 0; lane < 32; ++lane) {
          r.v[0][lane] = f.a;
          r.v[1][lane] = f.b;
          r.v[2][lane] = f.c;
        }
        std::array<uint32_t, 32> old;
        std::copy(r.v[dst], r.v[dst] + 32, old.begin());
        ASSERT_EQ(fn(GOC_SEMANTICS_EXACT | GOC_SEMANTICS_STRICT, 0xaaaaaaaa, 0, r.v + dst, r.v,
                     r.v + 1, r.v + 2),
                  0);
        for (int lane = 0; lane < 32; ++lane)
          EXPECT_EQ(r.v[dst][lane], lane % 2 ? f.expected : old[lane]);
      }
  };
  check(kGfx12DotF16Cases, goc_rdna4_v_dot2_f32_f16);
  check(kGfx12DotBF16Cases, goc_rdna4_v_dot2_f32_bf16);
}
TEST(Wmma, AllModifierCombinations) {
  for (bool bf16 : {false, true})
    for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT})
      for (uint32_t modifiers = 0; modifiers < 64; ++modifiers) {
        Registers r;
        uint16_t one = bf16 ? 0x3f80 : 0x3c00;
        for (int reg = 0; reg < 8; ++reg)
          for (int lane = 0; lane < 32; ++lane)
            r.v[reg][lane] = one | (uint32_t(one) << 16);
        for (int reg = 8; reg < 16; ++reg)
          std::fill(r.v[reg], r.v[reg] + 32, goc::as_bits(-3.0f));
        int low = ((modifiers & 1) != 0) ^ ((modifiers & 2) != 0) ? -8 : 8;
        int high = ((modifiers & 8) != 0) ^ ((modifiers & 16) != 0) ? -8 : 8;
        int c = (modifiers & GOC_WMMA_ABS_C) ? 3 : -3;
        if (modifiers & GOC_WMMA_NEG_C)
          c = -c;
        auto fn = bf16 ? goc_rdna4_v_wmma_f32_16x16x16_bf16 : goc_rdna4_v_wmma_f32_16x16x16_f16;
        ASSERT_EQ(fn(semantics | GOC_SEMANTICS_STRICT, UINT32_MAX, modifiers, r.v + 8, r.v, r.v + 4,
                     r.v + 8),
                  0);
        for (int reg = 8; reg < 16; ++reg)
          for (int lane = 0; lane < 32; ++lane)
            EXPECT_EQ(r.v[reg][lane], goc::as_bits(float(low + high + c)));
      }
}

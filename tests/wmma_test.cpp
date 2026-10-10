// SPDX-License-Identifier: MIT

#include "dense_golden.h"
#include "dot.h"
#include "dot_fixtures.h"
#include "float_formats.h"
#include "goc/goc.h"
#include "internal.h"
#include "test_instruction.h"
#include "wmma_fixtures.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <initializer_list>
#include <random>
#include <stdint.h>
#include <vector>

namespace {

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
  for (uint64_t level = 0; level <= goc_init_cpu_flags(); ++level)
    for (bool bf16 : {false, true}) {
      auto fn = bf16 ? goc_v_wmma_f32_16x16x16_bf16_rdna4 : goc_v_wmma_f32_16x16x16_f16_rdna4;
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
          ASSERT_EQ(fn(level, 0, r.v + dst, r.v, r.v + 4, r.v + 8), 0);
          for (int reg = 0; reg < 8; ++reg)
            for (int lane = 0; lane < 32; ++lane)
              EXPECT_TRUE(fuzzy(goc::as_float(r.v[dst + reg][lane]), goc::as_float(f.expected32)));
          r.guards();
        }
    }
}

TEST(Wmma, LaneMappingAndPartialOperandOverlap) {
  for (uint64_t level = 0; level <= goc_init_cpu_flags(); ++level)
    for (bool bf16 : {false, true}) {
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
      auto fn = bf16 ? goc_v_wmma_f32_16x16x16_bf16_rdna4 : goc_v_wmma_f32_16x16x16_f16_rdna4;
      ASSERT_EQ(fn(level, 0, r.v + 2, r.v, r.v + 4, r.v + 8), 0);
      for (int lane = 0; lane < 32; ++lane)
        for (int reg = 0; reg < 8; ++reg) {
          int row = reg + 8 * (lane / 16), col = lane % 16;
          uint32_t expected = goc::as_bits(std::ldexp(1.0f, (row + col) % 8));
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
  EXPECT_EQ(goc_v_wmma_f32_16x16x16_f16_rdna4(GOC_SEMANTICS_MASK | GOC_SEMANTICS_STRICT, 0, r.v,
                                              r.v, r.v, r.v),
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
  for (uint64_t level = 0; level <= goc_init_cpu_flags(); ++level)
    for (bool bf16 : {false, true}) {
      auto fn = bf16 ? goc_v_wmma_f32_16x16x16_bf16_rdna4 : goc_v_wmma_f32_16x16x16_f16_rdna4;
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
          ASSERT_EQ(fn(level | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, r.v + dst,
                       r.v, r.v + 4, r.v + 8),
                    0);
          for (int reg = 0; reg < 8; ++reg)
            for (int lane = 0; lane < 32; ++lane)
              EXPECT_EQ(r.v[dst + reg][lane], f.expected32);
          r.guards();
        }
    }
}

TEST(Dot, HardwareCapturedSpecialValuesAndRounding) {
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

TEST(Dot, PublicApiHardwareFixturesMaskAndOverlap) {
  const auto check = [](const auto &cases, auto fn) {
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
        ASSERT_EQ(goc_test::without_exceptions(fn,
                                               GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                                               0xaaaaaaaa, 0, r.v + dst, r.v, r.v + 1, r.v + 2),
                  0);
        for (int lane = 0; lane < 32; ++lane)
          EXPECT_EQ(r.v[dst][lane], lane % 2 ? f.expected : old[lane]);
      }
  };
  check(kGfx12DotF16Cases, goc_v_dot2_f32_f16_rdna4);
  check(kGfx12DotBF16Cases, goc_v_dot2_f32_bf16_rdna4);
}

TEST(Wmma, AllModifierCombinations) {
  for (bool bf16 : {false, true})
    for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
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
        auto fn = bf16 ? goc_v_wmma_f32_16x16x16_bf16_rdna4 : goc_v_wmma_f32_16x16x16_f16_rdna4;
        ASSERT_EQ(fn(semantics | GOC_SEMANTICS_STRICT, modifiers, r.v + 8, r.v, r.v + 4, r.v + 8),
                  0);
        for (int reg = 8; reg < 16; ++reg)
          for (int lane = 0; lane < 32; ++lane)
            EXPECT_EQ(r.v[reg][lane], goc::as_bits(float(low + high + c)));
      }
}

TEST(Wmma, Bf16FastPathSubnormalFallback) {
  Registers r;

  // 16 * min-normal * 0.5 is a normal result, despite subnormal products.
  // Include a subnormal C to require the scalar fallback.
  for (int reg = 0; reg < 4; ++reg)
    for (int lane = 0; lane < 32; ++lane) {
      r.v[reg][lane] = 0x00800080;
      r.v[4 + reg][lane] = 0x3f003f00;
    }
  for (int reg = 8; reg < 16; ++reg)
    std::fill(r.v[reg], r.v[reg] + 32, 1u);
  ASSERT_EQ(goc_v_wmma_f32_16x16x16_bf16_rdna4(0, 0, r.v + 16, r.v, r.v + 4, r.v + 8), 0);
  ASSERT_EQ(
      goc_v_wmma_f32_16x16x16_bf16_rdna4(goc_init_cpu_flags(), 0, r.v + 8, r.v, r.v + 4, r.v + 8),
      0);
  for (int reg = 0; reg < 8; ++reg)
    for (int lane = 0; lane < 32; ++lane)
      EXPECT_EQ(r.v[8 + reg][lane], r.v[16 + reg][lane]);
}

TEST(Wmma, NormalBf16FactorsWithSubnormalProducts) {
  Registers r;
  for (int reg = 0; reg < 4; ++reg)
    for (int lane = 0; lane < 32; ++lane) {
      r.v[reg][lane] = 0x00800080;
      r.v[4 + reg][lane] = 0x3f003f00;
    }
  ASSERT_EQ(
      goc_v_wmma_f32_16x16x16_bf16_rdna4(goc_init_cpu_flags(), 0, r.v + 16, r.v, r.v + 4, r.v + 8),
      0);
  for (int reg = 16; reg < 24; ++reg)
    for (int lane = 0; lane < 32; ++lane)
      EXPECT_EQ(r.v[reg][lane], 0x02000000u);
}

TEST(Wmma, DeterministicDenseIntegerGolden) {
  for (uint64_t level = 0; level <= goc_init_cpu_flags(); ++level)
    for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
      for (bool bf16 : {false, true})
        for (int dst : {0, 2, 8, 16}) {
          Registers r;
          std::minstd_rand rng(7);
          const uint16_t f16[] = {0xc000, 0xbc00, 0, 0x3c00, 0x4000};
          const uint16_t b16[] = {0xc000, 0xbf80, 0, 0x3f80, 0x4000};
          const auto *values = bf16 ? b16 : f16;
          for (int row = 0; row < 16; ++row)
            for (int k = 0; k < 16; ++k)
              set16(r.v, row, k, values[rng() % 5]);
          for (int k = 0; k < 16; ++k)
            for (int col = 0; col < 16; ++col)
              set16(r.v + 4, col, k, values[rng() % 5]);
          for (int row = 0; row < 16; ++row)
            for (int col = 0; col < 16; ++col)
              r.v[8 + row % 8][col + 16 * (row / 8)] = goc::as_bits(float(int(rng() % 5) - 2));
          auto fn = bf16 ? goc_v_wmma_f32_16x16x16_bf16_rdna4 : goc_v_wmma_f32_16x16x16_f16_rdna4;
          ASSERT_EQ(
              fn(level | semantics | GOC_SEMANTICS_STRICT, 0, r.v + dst, r.v, r.v + 4, r.v + 8), 0);
          for (int row = 0; row < 16; ++row)
            for (int col = 0; col < 16; ++col)
              EXPECT_EQ(r.v[dst + row % 8][col + 16 * (row / 8)],
                        goc::as_bits(float(kDenseGolden[row * 16 + col])))
                  << "row=" << row << " col=" << col << " level=" << level
                  << " semantics=" << semantics;
        }
}

TEST(Dot, LooseAndFallbackIntegerGolden) {
  for (bool bf16 : {false, true})
    for (uint64_t semantics :
         {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL, GOC_SEMANTICS_MASK}) {
      Registers r;
      uint32_t a = bf16 ? 0x40003f80u : 0x40003c00u; // 1,2
      uint32_t b = bf16 ? 0x40804040u : 0x44004200u; // 3,4
      for (int lane = 0; lane < 32; ++lane) {
        r.v[0][lane] = a;
        r.v[1][lane] = b;
        r.v[2][lane] = 0xbf800000;
      }
      auto fn = bf16 ? goc_v_dot2_f32_bf16_rdna4 : goc_v_dot2_f32_f16_rdna4;
      ASSERT_EQ(fn(semantics, UINT32_MAX, 0, r.v + 2, r.v, r.v + 1, r.v + 2, nullptr), 0);
      for (int lane = 0; lane < 32; ++lane)
        EXPECT_EQ(r.v[2][lane], 0x41200000u); // 1*3+2*4-1=10
      EXPECT_EQ(fn(GOC_SEMANTICS_MASK | GOC_SEMANTICS_STRICT, UINT32_MAX, 0, r.v + 2, r.v, r.v + 1,
                   r.v + 2, nullptr),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
      for (int lane = 0; lane < 32; ++lane)
        EXPECT_EQ(r.v[2][lane], 0x41200000u);
    }
}

TEST(WmmaWave64, CapturedGoldensAndOverlap) {
  for (bool bf16 : {false, true})
    for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL}) {
      auto fn = bf16 ? goc_v_wmma_f32_16x16x16_bf16_rdna4_wave64
                     : goc_v_wmma_f32_16x16x16_f16_rdna4_wave64;
      const auto &cases = bf16 ? kGfx12WmmaBF16Cases : kGfx12WmmaF16Cases;
      for (const auto &f : cases)
        for (int dst : {0, 1, 2, 4, 8}) {
          uint32_t data[12][64] = {};
          uint32_t *v[12];
          for (int i = 0; i < 12; ++i)
            v[i] = data[11 - i];
          for (int reg = 0; reg < 2; ++reg)
            for (int lane = 0; lane < 64; ++lane) {
              int k = 4 * (lane / 16) + 2 * reg;
              v[reg][lane] = f.a[k] | (uint32_t(f.a[k + 1]) << 16);
              v[2 + reg][lane] = f.b[k] | (uint32_t(f.b[k + 1]) << 16);
            }
          for (int reg = 4; reg < 8; ++reg)
            std::fill(v[reg], v[reg] + 64, f.c);
          ASSERT_EQ(fn(semantics | GOC_SEMANTICS_STRICT, 0, v + dst, v, v + 2, v + 4), 0);
          for (int reg = 0; reg < 4; ++reg)
            for (int lane = 0; lane < 64; ++lane) {
              if (semantics == GOC_SEMANTICS_EXACT_EMPIRICAL)
                EXPECT_EQ(v[dst + reg][lane], f.expected64);
              else
                EXPECT_TRUE(fuzzy(goc::as_float(v[dst + reg][lane]), goc::as_float(f.expected64)));
            }
        }
    }
}

TEST(WmmaWave64, DenseLaneMapping) {
  for (bool bf16 : {false, true})
    for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL}) {
      uint32_t data[12][64] = {};
      uint32_t *v[12];
      for (int i = 0; i < 12; ++i)
        v[i] = data[i];
      std::minstd_rand rng(7);
      const uint16_t f16[] = {0xc000, 0xbc00, 0, 0x3c00, 0x4000};
      const uint16_t b16[] = {0xc000, 0xbf80, 0, 0x3f80, 0x4000};
      auto values = bf16 ? b16 : f16;
      for (int row = 0; row < 16; ++row)
        for (int k = 0; k < 16; ++k)
          v[k % 4 / 2][row + 16 * (k / 4)] |= uint32_t(values[rng() % 5]) << (16 * (k % 2));
      for (int k = 0; k < 16; ++k)
        for (int col = 0; col < 16; ++col)
          v[2 + k % 4 / 2][col + 16 * (k / 4)] |= uint32_t(values[rng() % 5]) << (16 * (k % 2));
      for (int row = 0; row < 16; ++row)
        for (int col = 0; col < 16; ++col)
          v[4 + row % 4][col + 16 * (row / 8) + 32 * ((row / 4) % 2)] =
              goc::as_bits(float(int(rng() % 5) - 2));
      auto fn = bf16 ? goc_v_wmma_f32_16x16x16_bf16_rdna4_wave64
                     : goc_v_wmma_f32_16x16x16_f16_rdna4_wave64;
      ASSERT_EQ(fn(semantics | GOC_SEMANTICS_STRICT, 0, v + 8, v, v + 2, v + 4), 0);
      for (int row = 0; row < 16; ++row)
        for (int col = 0; col < 16; ++col)
          EXPECT_EQ(v[8 + row % 4][col + 16 * (row / 8) + 32 * ((row / 4) % 2)],
                    goc::as_bits(float(kDenseGolden[row * 16 + col])));
    }
}

TEST(WmmaWave64, ModifiersAndErrorsPreserveState) {
  for (bool bf16 : {false, true})
    for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
      for (uint32_t modifiers = 0; modifiers < 64; ++modifiers) {
        uint32_t data[8][64];
        uint32_t *v[8];
        for (int i = 0; i < 8; ++i)
          v[i] = data[i];
        uint32_t one = bf16 ? 0x3f80u : 0x3c00u;
        for (int reg = 0; reg < 4; ++reg)
          std::fill(v[reg], v[reg] + 64, one | (one << 16));
        for (int reg = 4; reg < 8; ++reg)
          std::fill(v[reg], v[reg] + 64, 0xc0400000u);
        auto fn = bf16 ? goc_v_wmma_f32_16x16x16_bf16_rdna4_wave64
                       : goc_v_wmma_f32_16x16x16_f16_rdna4_wave64;
        ASSERT_EQ(fn(semantics | GOC_SEMANTICS_STRICT, 64, v + 4, v, v + 2, v + 4),
                  GOC_ERROR_INVALID_FLAGS);
        for (int reg = 4; reg < 8; ++reg)
          for (int lane = 0; lane < 64; ++lane)
            ASSERT_EQ(v[reg][lane], 0xc0400000u);
        int low = ((modifiers & 1) != 0) ^ ((modifiers & 2) != 0) ? -8 : 8;
        int high = ((modifiers & 8) != 0) ^ ((modifiers & 16) != 0) ? -8 : 8;
        int c = (modifiers & GOC_WMMA_ABS_C) ? 3 : -3;
        if (modifiers & GOC_WMMA_NEG_C)
          c = -c;
        ASSERT_EQ(fn(semantics | GOC_SEMANTICS_STRICT, modifiers, v + 4, v, v + 2, v + 4), 0);
        for (int reg = 4; reg < 8; ++reg)
          for (int lane = 0; lane < 64; ++lane)
            EXPECT_EQ(v[reg][lane], goc::as_bits(float(low + high + c)));
      }
}

TEST(Wmma, Fp16V3FiniteAndExceptionalInputsMatchScalar) {
  if (goc_init_cpu_flags() < GOC_CPU_X86_64_V3)
    GTEST_SKIP() << "Host does not support x86-64-v3";
  for (bool exceptional : {false, true})
    for (int trial = 0; trial < 8; ++trial)
      for (int dst : {0, 2, 4, 8, 16}) {
        Registers reference, actual;
        std::minstd_rand random(73 + trial);
        for (int reg = 0; reg < 24; ++reg)
          for (int lane = 0; lane < 32; ++lane) {
            uint32_t bits = uint32_t(random()) ^ (uint32_t(random()) << 16);
            if (reg < 8) {
              // Both signs and the full finite exponent range, including
              // FP16 subnormals. Separate trials inject NaNs and infinities.
              bits &= 0xfbfffbff;
              if (exceptional && lane == 3 && reg == 0)
                bits = 0x7c00fc00;
              if (exceptional && lane == 11 && reg == 5)
                bits = 0x7e557d23;
            } else {
              bits &= 0xff7fffff;
            }
            reference.v[reg][lane] = actual.v[reg][lane] = bits;
          }
        ASSERT_EQ(goc_v_wmma_f32_16x16x16_f16_rdna4(GOC_CPU_BASELINE, 0, reference.v + dst,
                                                    reference.v, reference.v + 4, reference.v + 8),
                  GOC_SUCCESS);
        ASSERT_EQ(goc_v_wmma_f32_16x16x16_f16_rdna4(GOC_CPU_X86_64_V3, 0, actual.v + dst, actual.v,
                                                    actual.v + 4, actual.v + 8),
                  GOC_SUCCESS);
        // Finite values use the same K order and FMA operations. Loose NaN
        // payloads may differ between the CPU converter and scalar widening.
        for (int reg = 0; reg < 24; ++reg)
          for (int lane = 0; lane < 32; ++lane) {
            bool written = reg >= dst && reg < dst + 8;
            if (written && std::isnan(goc::as_float(reference.v[reg][lane])))
              EXPECT_TRUE(std::isnan(goc::as_float(actual.v[reg][lane])));
            else
              EXPECT_EQ(actual.v[reg][lane], reference.v[reg][lane]);
          }
        reference.guards();
        actual.guards();
      }
}

TEST(Wmma, Bf16V3AndZen4FallbackMatchScalar) {
  if (goc_init_cpu_flags() < GOC_CPU_X86_64_V3)
    GTEST_SKIP() << "Host does not support x86-64-v3";
  for (uint64_t level = GOC_CPU_X86_64_V3; level <= goc_init_cpu_flags(); ++level)
    for (int trial = 0; trial < 8; ++trial)
      for (int dst : {0, 2, 4, 8, 16}) {
        Registers reference, actual;
        std::minstd_rand random(107 + trial);
        for (int reg = 0; reg < 24; ++reg)
          for (int lane = 0; lane < 32; ++lane) {
            uint32_t bits = uint32_t(random()) ^ (uint32_t(random()) << 16);
            if (reg < 8) {
              // Alternate moderate finite inputs with the full BF16 range.
              if (trial % 2 == 0)
                bits = (bits & 0x807f807f) | 0x3f003f00;
              if (trial == 3 && reg == 0 && lane == 3)
                bits = 0x7f80ff80;
              if (trial == 5 && reg == 5 && lane == 11)
                bits = 0x7fc57f83;
              // A subnormal forces Zen4 to reject DPBF16 and use its fallback.
              if (reg == 0 && lane == 0)
                bits = 0x00010001;
            } else {
              bits &= 0xff7fffff;
            }
            reference.v[reg][lane] = actual.v[reg][lane] = bits;
          }
        ASSERT_EQ(goc_v_wmma_f32_16x16x16_bf16_rdna4(GOC_CPU_BASELINE, 0, reference.v + dst,
                                                     reference.v, reference.v + 4, reference.v + 8),
                  GOC_SUCCESS);
        ASSERT_EQ(goc_v_wmma_f32_16x16x16_bf16_rdna4(level, 0, actual.v + dst, actual.v,
                                                     actual.v + 4, actual.v + 8),
                  GOC_SUCCESS);
        for (int reg = 0; reg < 24; ++reg)
          for (int lane = 0; lane < 32; ++lane) {
            bool written = reg >= dst && reg < dst + 8;
            if (written && std::isnan(goc::as_float(reference.v[reg][lane])))
              EXPECT_TRUE(std::isnan(goc::as_float(actual.v[reg][lane])));
            else
              EXPECT_EQ(actual.v[reg][lane], reference.v[reg][lane]);
          }
        reference.guards();
        actual.guards();
      }
}

TEST(Wmma, Bf16V3BoundaryGoldens) {
  struct Case {
    uint16_t a, b;
    uint32_t c, expected;
  };

  // Uniform factors: D = 16*A*B+C. These literal goldens exercise conditions
  // that DPBF16 cannot handle with ordinary host FMA semantics.
  const Case cases[] = {
      {0x0001, 0x3f80, 0, 0x00100000}, // Minimum BF16 subnormal times 16.
      {0x8001, 0x3f80, 0, 0x80100000},
      {0x0080, 0x3f00, 0, 0x02000000}, // Normal factors with subnormal products.
      {0x0080, 0x0080, 0, 0},          // Products underflow to zero.
      {0, 0x3f80, 1, 1},               // Preserve the smallest FP32 accumulator.
      {0x7f7f, 0x4000, 0, 0x7f800000}, // Finite factors overflow.
      {0xff80, 0x3f80, 0, 0xff800000},
  };
  for (uint64_t level = 0; level <= goc_init_cpu_flags(); ++level)
    for (const auto &f : cases) {
      Registers r;
      for (int reg = 0; reg < 8; ++reg)
        for (int lane = 0; lane < 32; ++lane) {
          if (reg < 4) {
            r.v[reg][lane] = uint32_t(f.a) | (uint32_t(f.a) << 16);
            r.v[4 + reg][lane] = uint32_t(f.b) | (uint32_t(f.b) << 16);
          }
          r.v[8 + reg][lane] = f.c;
        }
      ASSERT_EQ(goc_v_wmma_f32_16x16x16_bf16_rdna4(level, 0, r.v + 16, r.v, r.v + 4, r.v + 8),
                GOC_SUCCESS);
      for (int reg = 16; reg < 24; ++reg)
        for (int lane = 0; lane < 32; ++lane)
          EXPECT_EQ(r.v[reg][lane], f.expected);
    }
}

TEST(Wmma, Bf16CpuLevelsRespectModifiers) {
  for (uint64_t level = 0; level <= goc_init_cpu_flags(); ++level)
    for (uint32_t modifiers = 0; modifiers < 64; ++modifiers) {
      Registers r;
      for (int reg = 0; reg < 8; ++reg)
        for (int lane = 0; lane < 32; ++lane) {
          if (reg < 4) {
            r.v[reg][lane] = 0x40003f80;     // Low A=1, high A=2.
            r.v[4 + reg][lane] = 0x40804040; // Low B=3, high B=4.
          }
          r.v[8 + reg][lane] = goc::as_bits(-5.0f);
        }
      int low = (bool(modifiers & 1) != bool(modifiers & 2)) ? -24 : 24;
      int high = (bool(modifiers & 8) != bool(modifiers & 16)) ? -64 : 64;
      int c = (modifiers & GOC_WMMA_ABS_C) ? 5 : -5;
      if (modifiers & GOC_WMMA_NEG_C)
        c = -c;
      ASSERT_EQ(
          goc_v_wmma_f32_16x16x16_bf16_rdna4(level, modifiers, r.v + 8, r.v, r.v + 4, r.v + 8),
          GOC_SUCCESS);
      for (int reg = 8; reg < 16; ++reg)
        for (int lane = 0; lane < 32; ++lane)
          EXPECT_EQ(r.v[reg][lane], goc::as_bits(float(low + high + c)));
    }
}

TEST(Wmma, Bf16Zen4EligibilityChecksEveryPackedElement) {
  if (goc_init_cpu_flags() < GOC_CPU_ZEN4)
    GTEST_SKIP() << "Host does not support the Zen4 feature set";

  struct FactorCase {
    uint16_t value, partner;
  };

  const FactorCase cases[] = {
      {1, 0x3f80}, {0x8001, 0x3f80}, {0x0080, 0x3f00}, {0x7f80, 0x3f80}, {0x7fc1, 0x3f80}};
  for (int operand = 0; operand < 2; ++operand)
    for (int reg = 0; reg < 4; ++reg)
      for (int lane = 0; lane < 32; ++lane)
        for (int half = 0; half < 2; ++half)
          for (const auto &f : cases) {
            Registers reference, actual;
            for (int other_reg = 0; other_reg < 4; ++other_reg)
              for (int other_lane = 0; other_lane < 32; ++other_lane) {
                uint32_t word = uint32_t(f.partner) | (uint32_t(f.partner) << 16);
                reference.v[4 * (1 - operand) + other_reg][other_lane] = word;
                actual.v[4 * (1 - operand) + other_reg][other_lane] = word;
              }
            reference.v[4 * operand + reg][lane] = uint32_t(f.value) << (16 * half);
            actual.v[4 * operand + reg][lane] = uint32_t(f.value) << (16 * half);
            ASSERT_EQ(goc_v_wmma_f32_16x16x16_bf16_rdna4(GOC_CPU_BASELINE, 0, reference.v + 16,
                                                         reference.v, reference.v + 4,
                                                         reference.v + 8),
                      GOC_SUCCESS);
            ASSERT_EQ(goc_v_wmma_f32_16x16x16_bf16_rdna4(GOC_CPU_ZEN4, 0, actual.v + 16, actual.v,
                                                         actual.v + 4, actual.v + 8),
                      GOC_SUCCESS);
            // All other products are zero, so the selected subnormal product
            // survives in the output. An incorrect DPBF16 selection flushes it.
            for (int out_reg = 16; out_reg < 24; ++out_reg)
              for (int out_lane = 0; out_lane < 32; ++out_lane) {
                float expected = goc::as_float(reference.v[out_reg][out_lane]);
                if (std::isnan(expected))
                  EXPECT_TRUE(std::isnan(goc::as_float(actual.v[out_reg][out_lane])));
                else
                  EXPECT_EQ(actual.v[out_reg][out_lane], reference.v[out_reg][out_lane]);
              }
          }
}

TEST(Wmma, Bf16Zen4EligibilityChecksEveryAccumulator) {
  if (goc_init_cpu_flags() < GOC_CPU_ZEN4)
    GTEST_SKIP() << "Host does not support the Zen4 feature set";
  for (int reg = 0; reg < 8; ++reg)
    for (int lane = 0; lane < 32; ++lane)
      for (uint32_t bits : {1u, 0x80000001u, 0x7f800000u, 0x7fc12345u}) {
        Registers r;
        r.v[8 + reg][lane] = bits;
        ASSERT_EQ(
            goc_v_wmma_f32_16x16x16_bf16_rdna4(GOC_CPU_ZEN4, 0, r.v + 16, r.v, r.v + 4, r.v + 8),
            GOC_SUCCESS);
        for (int out_reg = 0; out_reg < 8; ++out_reg)
          for (int out_lane = 0; out_lane < 32; ++out_lane) {
            uint32_t expected = out_reg == reg && out_lane == lane ? bits : 0;
            if (std::isnan(goc::as_float(expected)))
              EXPECT_TRUE(std::isnan(goc::as_float(r.v[16 + out_reg][out_lane])));
            else
              EXPECT_EQ(r.v[16 + out_reg][out_lane], expected);
          }
      }
}

TEST(Wmma, SimdModifiersDenseReferenceAndOverlap) {
  const uint16_t fp16[] = {0xc000, 0xbc00, 0, 0x3c00, 0x4000};
  const uint16_t bf16[] = {0xc000, 0xbf80, 0, 0x3f80, 0x4000};
  for (bool bf : {false, true})
    for (uint32_t modifiers = 0; modifiers < 64; ++modifiers)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (int dst : {0, 2, 4, 8, 16}) {
          SCOPED_TRACE(::testing::Message() << "bf16=" << bf << " modifiers=" << modifiers
                                            << " cpu=" << cpu << " dst=" << dst);
          Registers r;
          int a[16][16], b[16][16], c[16][16];
          std::minstd_rand random(87);
          for (int row = 0; row < 16; ++row)
            for (int k = 0; k < 16; ++k) {
              int ai = random() % 5, bi = random() % 5;
              a[row][k] = ai - 2;
              b[row][k] = bi - 2;
              set16(r.v, row, k, bf ? bf16[ai] : fp16[ai]);
              set16(r.v + 4, row, k, bf ? bf16[bi] : fp16[bi]);
            }
          for (int row = 0; row < 16; ++row)
            for (int col = 0; col < 16; ++col) {
              c[row][col] = int(random() % 9) - 4;
              r.v[8 + row % 8][col + 16 * (row / 8)] = goc::as_bits(float(c[row][col]));
            }
          uint32_t before[24][32];
          for (int reg = 0; reg < 24; ++reg)
            std::copy(r.v[reg], r.v[reg] + 32, before[reg]);
          auto fn = bf ? goc_v_wmma_f32_16x16x16_bf16_rdna4 : goc_v_wmma_f32_16x16x16_f16_rdna4;
          ASSERT_EQ(fn(cpu, modifiers, r.v + dst, r.v, r.v + 4, r.v + 8), GOC_SUCCESS);
          for (int row = 0; row < 16; ++row)
            for (int col = 0; col < 16; ++col) {
              int acc = (modifiers & GOC_WMMA_ABS_C) ? std::abs(c[row][col]) : c[row][col];
              if (modifiers & GOC_WMMA_NEG_C)
                acc = -acc;
              for (int k = 0; k < 16; ++k) {
                int left = a[row][k], right = b[col][k];
                if (modifiers & (k % 2 ? GOC_WMMA_NEG_HI_A : GOC_WMMA_NEG_LO_A))
                  left = -left;
                if (modifiers & (k % 2 ? GOC_WMMA_NEG_HI_B : GOC_WMMA_NEG_LO_B))
                  right = -right;
                acc += left * right;
              }
              int reg = dst + row % 8, lane = col + 16 * (row / 8);
              {
                EXPECT_EQ(goc::as_float(r.v[reg][lane]), float(acc));
              }
            }
          for (int reg = 0; reg < 24; ++reg)
            for (int lane = 0; lane < 32; ++lane)
              if (reg < dst || reg >= dst + 8) {
                EXPECT_EQ(r.v[reg][lane], before[reg][lane]);
              }
          r.guards();
        }
}

TEST(Wmma, SimdModifiersSpecialValuesAndFallback) {
  struct Case {
    uint16_t fp16_a, fp16_b, bf16_a, bf16_b;
    uint32_t c;
  };

  const Case cases[] = {
      {1, 0x3c00, 1, 0x3f80, 0}, // Subnormal factors.
      {0x8001, 0x3c00, 0x8001, 0x3f80, 0},
      {0x0400, 0x3800, 0x0080, 0x3f00, 0},          // BF16 normal factors, subnormal products.
      {0x7bff, 0x4000, 0x7f7f, 0x4000, 0},          // BF16 product overflow.
      {0x7c00, 0x3c00, 0x7f80, 0x3f80, 0},          // Infinity.
      {0x7e55, 0x3c00, 0x7fc5, 0x3f80, 0},          // Quiet NaN.
      {0x7d23, 0x3c00, 0x7f83, 0x3f80, 0},          // Signaling NaN.
      {0x8000, 0x3c00, 0x8000, 0x3f80, 0x80000000}, // Signed zeros.
      {0, 0x3c00, 0, 0x3f80, 1},                    // Subnormal accumulator.
      {0, 0x3c00, 0, 0x3f80, 0x80000001},
      {0, 0x3c00, 0, 0x3f80, 0xff800000}, // Infinite accumulator.
      {0, 0x3c00, 0, 0x3f80, 0xffc12345}, // NaN accumulator.
  };
  for (bool bf : {false, true})
    for (const auto &f : cases)
      for (uint32_t modifiers = 0; modifiers < 64; ++modifiers) {
        Registers reference;
        const auto fill = [&](Registers &r) {
          uint32_t a = bf ? f.bf16_a : f.fp16_a, b = bf ? f.bf16_b : f.fp16_b;
          for (int reg = 0; reg < 4; ++reg)
            for (int lane = 0; lane < 32; ++lane) {
              r.v[reg][lane] = a | (a << 16);
              r.v[4 + reg][lane] = b | (b << 16);
            }
          for (int reg = 8; reg < 16; ++reg)
            std::fill(r.v[reg], r.v[reg] + 32, f.c);
        };
        fill(reference);
        auto fn = bf ? goc_v_wmma_f32_16x16x16_bf16_rdna4 : goc_v_wmma_f32_16x16x16_f16_rdna4;
        ASSERT_EQ(fn(GOC_CPU_BASELINE, modifiers, reference.v + 16, reference.v, reference.v + 4,
                     reference.v + 8),
                  GOC_SUCCESS);
        for (uint64_t cpu = GOC_CPU_X86_64_V3; cpu <= goc_init_cpu_flags(); ++cpu) {
          SCOPED_TRACE(::testing::Message() << "bf16=" << bf << " modifiers=" << modifiers
                                            << " cpu=" << cpu << " a=" << f.bf16_a << " c=" << f.c);
          Registers actual;
          fill(actual);
          ASSERT_EQ(fn(cpu, modifiers, actual.v + 8, actual.v, actual.v + 4, actual.v + 8),
                    GOC_SUCCESS);
          for (int reg = 0; reg < 8; ++reg)
            for (int lane = 0; lane < 32; ++lane) {
              float want = goc::as_float(reference.v[16 + reg][lane]);
              if (std::isnan(want))
                EXPECT_TRUE(std::isnan(goc::as_float(actual.v[8 + reg][lane])));
              else
                EXPECT_EQ(actual.v[8 + reg][lane], reference.v[16 + reg][lane]);
            }
          actual.guards();
        }
      }
}

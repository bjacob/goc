// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "internal.h"
#include "rdna4_exec_masks.h"
#include "rdna4_fma_omod_hardware.h"
#include "rdna4_omod_reference.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <initializer_list>
#include <random>
#include <stdint.h>

extern "C" int goc_test_c_api(void);

TEST(Api, C99LinksAgainstCppImplementation) { EXPECT_EQ(goc_test_c_api(), 1); }

TEST(Arithmetic, ConstInputs) {
  const uint32_t a[32] = {0x40000000}, b[32] = {0x40400000}, c[32] = {0x40800000};
  const uint32_t *const pa = a, *const pb = b, *const pc = c;
  uint32_t d[32] = {};
  uint32_t *pd = d;
  ASSERT_EQ(goc_rdna4_v_fma_f32(0, 1, 0, &pd, &pa, &pb, &pc), GOC_SUCCESS);
  EXPECT_EQ(d[0], 0x41200000u);
}

TEST(Arithmetic, DeterministicFmaMaskAndAliasing) {
  std::mt19937 rng(42);
  std::array<uint32_t, 32> a, b, c;
  std::array<uint32_t, 32> expected;
  for (int i = 0; i < 32; ++i) {
    int x = int(rng() % 129) - 64, y = int(rng() % 129) - 64, z = int(rng() % 129) - 64;
    a[i] = goc::as_bits(float(x));
    b[i] = goc::as_bits(float(y));
    c[i] = goc::as_bits(float(z));
    expected[i] = goc::as_bits(float(x * y + z));
  }
  const auto original = a;
  auto pa = a.data(), pb = b.data(), pc = c.data();
  ASSERT_EQ(goc_rdna4_v_fma_f32(0, 0xaaaaaaaa, 0, &pa, &pa, &pb, &pc), 0);
  for (int i = 0; i < 32; ++i)
    EXPECT_EQ(a[i], (i % 2) ? expected[i] : original[i]);
}

TEST(Arithmetic, ErrorsPreserveDestination) {
  uint32_t a[32] = {}, d[32];
  auto pa = a, pd = d;
  for (auto &v : d)
    v = 0xdeadbeef;
  EXPECT_EQ(goc_rdna4_v_fma_f32(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, ~UINT64_C(0),
                                0, &pd, &pa, &pa, &pa),
            GOC_ERROR_UNSUPPORTED_SEMANTICS);
  EXPECT_EQ(goc_rdna4_v_log_f32(0, ~UINT64_C(0), GOC_ALU_NEG_B, &pd, &pa), GOC_ERROR_INVALID_FLAGS);
  EXPECT_EQ(goc_rdna4_v_log_f32(UINT64_C(1) << 63, ~UINT64_C(0), 0, &pd, &pa),
            GOC_ERROR_INVALID_FLAGS);
  for (auto v : d)
    EXPECT_EQ(v, 0xdeadbeef);
  EXPECT_EQ(goc_rdna4_v_fma_f32(GOC_SEMANTICS_EXACT_EMPIRICAL, 0, 0, &pd, &pa, &pa, &pa), 0);
  for (auto v : d)
    EXPECT_EQ(v, 0xdeadbeef);
}

TEST(Arithmetic, LogPowersOfTwoAndHighMaskBits) {
  uint32_t a[32];
  for (int i = 0; i < 32; ++i)
    a[i] = goc::as_bits(std::ldexp(1.0f, i - 16));
  auto pa = a;
  ASSERT_EQ(goc_rdna4_v_log_f32(0, UINT64_C(0xffffffff00000000), 0, &pa, &pa), 0);
  EXPECT_EQ(a[0], goc::as_bits(std::ldexp(1.0f, -16)));
  ASSERT_EQ(goc_rdna4_v_log_f32(0, UINT32_MAX, 0, &pa, &pa), 0);
  for (int i = 0; i < 32; ++i)
    EXPECT_EQ(a[i], goc::as_bits(float(i - 16)));
}

TEST(Arithmetic, AllCpuLevelsFmaGoldenAndAliasing) {
  // Literal IEEE inputs/outputs cover cancellation, fused rounding, subnormals,
  // infinities and signed zero. The fused witness is (1+2^-23)*(1-2^-23)-1.
  const uint32_t a_bits[] = {0x3f800001, 0x00000001, 0x7f800000, 0x80000000,
                             0x3fc00000, 0xc0000000, 0x00800000, 0x3f800000};
  const uint32_t b_bits[] = {0x3f7ffffe, 0x3f800000, 0x40000000, 0x40000000,
                             0x40000000, 0x40400000, 0x3f000000, 0x3f800000};
  const uint32_t c_bits[] = {0xbf800000, 0x00000001, 0x00000000, 0x80000000,
                             0xbf800000, 0x40c00000, 0x00000000, 0xbf800000};
  const uint32_t golden[] = {0xa8800000, 0x00000002, 0x7f800000, 0x80000000,
                             0x40000000, 0x00000000, 0x00400000, 0x00000000};
  for (uint64_t level = 0; level <= goc_init_cpu_flags(); ++level)
    for (int alias = 0; alias < 4; ++alias)
      for (uint64_t mask : rdna4_exec_masks()) {
        SCOPED_TRACE(::testing::Message()
                     << "level=" << level << " alias=" << alias << " mask=" << mask);
        uint32_t storage[4][34]; // Offsets avoid requiring SIMD alignment.
        uint32_t *ptrs[4];
        for (int j = 0; j < 4; ++j) {
          ptrs[j] = storage[j] + 1;
          storage[j][0] = storage[j][33] = 0xdeadbeef;
        }
        for (int i = 0; i < 32; ++i) {
          ptrs[0][i] = a_bits[i % 8];
          ptrs[1][i] = b_bits[i % 8];
          ptrs[2][i] = c_bits[i % 8];
          ptrs[3][i] = 0x12345678;
        }
        std::array<uint32_t, 32> before;
        std::copy(ptrs[alias], ptrs[alias] + 32, before.begin());
        ASSERT_EQ(goc_rdna4_v_fma_f32(level, mask, 0, &ptrs[alias], &ptrs[0], &ptrs[1], &ptrs[2]),
                  0);
        for (int i = 0; i < 32; ++i)
          EXPECT_EQ(ptrs[alias][i], ((mask >> i) & 1) ? golden[i % 8] : before[i]);
        for (int j = 0; j < 4; ++j) {
          EXPECT_EQ(storage[j][0], 0xdeadbeef);
          EXPECT_EQ(storage[j][33], 0xdeadbeef);
        }
      }
}

namespace {

void check_fma_modifiers(bool dx9) {
  auto fn = dx9 ? goc_rdna4_v_fma_dx9_zero_f32 : goc_rdna4_v_fma_f32;
  const float scales[] = {1, 2, 4, 0.5f};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (uint32_t modifiers = 0; modifiers < 512; ++modifiers)
      for (uint64_t mask : rdna4_exec_masks())
        for (int alias = 0; alias < 4; ++alias) {
          SCOPED_TRACE(::testing::Message()
                       << cpu << "/" << modifiers << "/" << mask << "/" << alias);
          uint32_t storage[4][34], original[32], expected[32];
          uint32_t *v[4];
          for (int reg = 0; reg < 4; ++reg) {
            std::fill(storage[reg], storage[reg] + 34, 0xdeadbeef);
            v[reg] = storage[reg] + 1;
          }
          for (int lane = 0; lane < 32; ++lane) {
            double inputs[3] = {(lane - 17) * 0.25, (lane % 7 - 3) * 0.5, (lane % 11 - 5) * 0.25};
            for (int source = 0; source < 3; ++source) {
              v[source][lane] = goc::as_bits(float(inputs[source]));
              if (modifiers & (8u << source))
                inputs[source] = std::abs(inputs[source]);
              if (modifiers & (1u << source))
                inputs[source] = -inputs[source];
            }
            // Dyadic inputs have an exactly representable double-precision
            // product/sum, independent of the implementation's FP32 FMA.
            float want = float(inputs[0] * inputs[1] + inputs[2]);
            if (dx9 && (inputs[0] == 0 || inputs[1] == 0))
              want = float(inputs[2]);
            want = dx9 ? want * scales[modifiers >> 6 & 3]
                       : goc_test::omod_f32_reference(want, modifiers);
            if (modifiers & GOC_ALU_CLAMP)
              want = std::min(1.0f, std::max(0.0f, want));
            expected[lane] = goc::as_bits(want);
            original[lane] = v[alias][lane];
          }
          ASSERT_EQ(fn(cpu, mask, modifiers, &v[alias], &v[0], &v[1], &v[2]), GOC_SUCCESS);
          for (int lane = 0; lane < 32; ++lane)
            EXPECT_EQ(v[alias][lane], ((mask >> lane) & 1) ? expected[lane] : original[lane]);
          for (const auto &reg : storage) {
            EXPECT_EQ(reg[0], 0xdeadbeef);
            EXPECT_EQ(reg[33], 0xdeadbeef);
          }
        }
}

} // namespace

TEST(Arithmetic, FmaAllModifiersCpuLevelsMasksAndAliases) { check_fma_modifiers(false); }

TEST(Arithmetic, Dx9FmaAllModifiersCpuLevelsMasksAndAliases) { check_fma_modifiers(true); }

TEST(Arithmetic, FmaModifierSpecialValues) {
  const uint32_t a_bits[] = {0x3f800001, 0x7fc12345, 0x7f800000, 0x80000000,
                             0x00000001, 0x7f7fffff, 0x3fc00000, 0x3fc00000};
  const uint32_t b_bits[] = {0x3f7ffffe, 0x3f800000, 0x00000000, 0x40000000,
                             0x3f800000, 0x40000000, 0x3fc00000, 0x3fc00000};
  const uint32_t c_bits[] = {0xbf800000, 0, 0, 0x80000000, 1, 0, 0xbe800000, 0xbe800000};
  const uint32_t modes[] = {
      GOC_ALU_OMOD_2,    GOC_ALU_CLAMP, GOC_ALU_CLAMP, GOC_ALU_OMOD_HALF,
      GOC_ALU_OMOD_HALF, GOC_ALU_CLAMP, GOC_ALU_ABS_C, GOC_ALU_ABS_C | GOC_ALU_NEG_C};
  const uint32_t golden[] = {0xa9000000, 0, 0, 0, 0, 0x3f800000, 0x40200000, 0x40000000};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (int test = 0; test < 8; ++test) {
      uint32_t a[32], b[32], c[32], d[32];
      std::fill(a, a + 32, a_bits[test]);
      std::fill(b, b + 32, b_bits[test]);
      std::fill(c, c + 32, c_bits[test]);
      auto pa = a, pb = b, pc = c, pd = d;
      ASSERT_EQ(goc_rdna4_v_fma_f32(cpu, UINT32_MAX, modes[test], &pd, &pa, &pb, &pc), GOC_SUCCESS);
      for (uint32_t value : d)
        EXPECT_EQ(value, golden[test]);
    }
}

TEST(Arithmetic, Dx9FmaZeroSelectionWithEveryModifierAndAlias) {
  const uint32_t values[] = {0,          0x80000000, 1,          0x80000001, 0x3f800000, 0xbf800000,
                             0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc12345, 0xffc12345,
                             0x7f812345, 0xff812345, 0x3f000000, 0xbf000000};
  const double scales[] = {1, 2, 4, 0.5};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (uint32_t mode = 0; mode < 512; ++mode)
      for (bool reverse : {false, true})
        for (int alias = 0; alias < 4; ++alias) {
          SCOPED_TRACE(::testing::Message()
                       << cpu << "/" << mode << "/" << reverse << "/" << alias);
          uint32_t storage[4][32], expected[32];
          uint32_t *v[4] = {storage[0], storage[1], storage[2], storage[3]};
          for (int lane = 0; lane < 32; ++lane) {
            uint32_t zero = lane & 1 ? 0x80000000 : 0;
            uint32_t other = values[(lane * 5 + 3) % 16];
            v[0][lane] = reverse ? other : zero;
            v[1][lane] = reverse ? zero : other;
            v[2][lane] = values[lane % 16];
            uint32_t want = v[2][lane];
            if (mode & GOC_ALU_ABS_C)
              want &= 0x7fffffff;
            if (mode & GOC_ALU_NEG_C)
              want ^= 0x80000000;
            if (mode & GOC_ALU_OMOD_HALF) {
              if ((want & 0x7fffffff) > 0x7f800000)
                want |= 0x00400000;
              else
                want = goc::as_bits(float(double(goc::as_float(want)) * scales[(mode >> 6) & 3]));
            }
            if (mode & GOC_ALU_CLAMP) {
              float value = goc::as_float(want);
              want = !(value > 0) ? 0 : value > 1 ? 0x3f800000 : want;
            }
            expected[lane] = want;
          }
          ASSERT_EQ(
              goc_rdna4_v_fma_dx9_zero_f32(cpu, UINT32_MAX, mode, &v[alias], &v[0], &v[1], &v[2]),
              GOC_SUCCESS);
          for (int lane = 0; lane < 32; ++lane)
            EXPECT_EQ(v[alias][lane], expected[lane]);
        }
}

TEST(Arithmetic, Dx9FmaFusedRoundingAndValidation) {
  // The ordinary fused-rounding witness must survive the DX9 extension.
  uint32_t a[32], b[32], c[32], d[32];
  std::fill(a, a + 32, 0x3f800001);
  std::fill(b, b + 32, 0x3f7ffffe);
  std::fill(c, c + 32, 0xbf800000);
  auto pa = a, pb = b, pc = c, pd = d;
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    std::fill(d, d + 32, 0xdeadbeef);
    EXPECT_EQ(goc_rdna4_v_fma_dx9_zero_f32(cpu, UINT32_MAX, GOC_ALU_HIGH_C, &pd, &pa, &pb, &pc),
              GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(
        goc_rdna4_v_fma_dx9_zero_f32(cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                                     UINT32_MAX, 0, &pd, &pa, &pb, &pc),
        GOC_ERROR_UNSUPPORTED_SEMANTICS);
    for (uint32_t value : d)
      EXPECT_EQ(value, 0xdeadbeef);
    ASSERT_EQ(goc_rdna4_v_fma_dx9_zero_f32(cpu, UINT32_MAX, 0, &pd, &pa, &pb, &pc), GOC_SUCCESS);
    for (uint32_t value : d)
      EXPECT_EQ(value, 0xa8800000);
  }
}

TEST(Arithmetic, FmaAndFmacOmodHardwareBoundaries) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (bool fmac : {false, true})
      for (unsigned omod = 0; omod < 4; ++omod)
        for (unsigned clamp = 0; clamp < 2; ++clamp)
          for (unsigned neg = 0; neg < 2; ++neg)
            for (uint64_t mask : rdna4_exec_masks())
              for (int alias = 0; alias < (fmac ? 1 : 4); ++alias) {
                SCOPED_TRACE(::testing::Message()
                             << cpu << '/' << fmac << '/' << omod << '/' << clamp << '/' << neg
                             << '/' << mask << '/' << alias);
                uint32_t words[4][34];
                uint32_t *p[4];
                for (int reg = 0; reg < 4; ++reg) {
                  std::fill(words[reg], words[reg] + 34, 0xdeadbeef);
                  p[reg] = words[reg] + 1;
                }
                for (int lane = 0; lane < 32; ++lane) {
                  p[0][lane] = 0x80000000;
                  p[1][lane] = goc_test::fma_omod_inputs[lane];
                  p[2][lane] = 0x3f800000;
                  p[3][lane] = 0x80000000;
                }
                uint32_t before[32];
                std::copy(p[alias], p[alias] + 32, before);
                uint64_t mode = (omod << 6) | (clamp ? GOC_ALU_CLAMP : 0) | neg;
                int error =
                    fmac ? goc_rdna4_v_fmac_f32(cpu, mask, mode, &p[alias], &p[1], &p[2])
                         : goc_rdna4_v_fma_f32(cpu, mask, mode, &p[alias], &p[1], &p[2], &p[3]);
                ASSERT_EQ(error, GOC_SUCCESS);
                for (int lane = 0; lane < 32; ++lane) {
                  uint32_t want = (mask >> lane) & 1
                                      ? goc_test::fma_omod_hardware[omod][clamp][neg][lane]
                                      : before[lane];
                  if ((want & 0x7fffffff) > 0x7f800000)
                    EXPECT_GT(p[alias][lane] & 0x7fffffff, 0x7f800000u);
                  else
                    EXPECT_EQ(p[alias][lane], want);
                }
                for (const auto &reg : words) {
                  EXPECT_EQ(reg[0], 0xdeadbeefu);
                  EXPECT_EQ(reg[33], 0xdeadbeefu);
                }
              }
}

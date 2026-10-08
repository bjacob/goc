// SPDX-License-Identifier: MIT

#include "goc_common.h"
#include "goc_rdna4.h"
#include "internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <initializer_list>
#include <random>
#include <stdint.h>

extern "C" int goc_test_c_api(void);

TEST(Api, C99LinksAgainstCppImplementation) { EXPECT_EQ(goc_test_c_api(), 1); }

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
  EXPECT_EQ(goc_rdna4_v_fma_f32(GOC_SEMANTICS_EXACT | GOC_SEMANTICS_STRICT, ~UINT64_C(0), 0, &pd,
                                &pa, &pa, &pa),
            GOC_ERROR_UNSUPPORTED_SEMANTICS);
  EXPECT_EQ(goc_rdna4_v_log_f32(0, ~UINT64_C(0), 1, &pd, &pa), GOC_ERROR_INVALID_FLAGS);
  EXPECT_EQ(goc_rdna4_v_log_f32(UINT64_C(1) << 63, ~UINT64_C(0), 0, &pd, &pa),
            GOC_ERROR_INVALID_FLAGS);
  for (auto v : d)
    EXPECT_EQ(v, 0xdeadbeef);
  EXPECT_EQ(goc_rdna4_v_fma_f32(GOC_SEMANTICS_EXACT, 0, 0, &pd, &pa, &pa, &pa), 0);
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
      for (uint32_t mask : {0u, 1u, 0x80000000u, 0x55555555u, 0xffffffffu}) {
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

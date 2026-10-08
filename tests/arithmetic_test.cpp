// SPDX-License-Identifier: MIT
#include "internal.h"
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <random>
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

// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "internal.h"
#include "rdna4_exec_masks.h"
#include "rdna4_omod_reference.h"
#include "rdna4_unary_hardware.h"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_rdna4_v_log_f32);
const Fn functions[] = {goc_rdna4_v_trunc_f32, goc_rdna4_v_ceil_f32,      goc_rdna4_v_rndne_f32,
                        goc_rdna4_v_floor_f32, goc_rdna4_v_sqrt_f32,      goc_rdna4_v_rcp_f32,
                        goc_rdna4_v_rsq_f32,   goc_rdna4_v_exp_f32,       goc_rdna4_v_log_f32,
                        goc_rdna4_v_fract_f32, goc_rdna4_v_frexp_mant_f32};

// Independent higher-precision reference, with explicit ties-to-even and
// binary32 rounding before output scaling.
float reference(int op, float input, uint32_t flags) {
  double x = input;
  if (flags & GOC_ALU_ABS_A)
    x = std::abs(x);
  if (flags & GOC_ALU_NEG_A)
    x = -x;
  bool flush = op >= 4 && op <= 8;
  if (flush && std::abs(x) < std::ldexp(1.0, -126))
    x = std::copysign(0.0, x);
  double y = 0;
  switch (op) {
  case 10: {
    int exponent;
    y = std::isfinite(x) ? std::frexp(x, &exponent) : x;
    break;
  }
  case 0:
    y = std::trunc(x);
    break;
  case 1:
    y = std::ceil(x);
    break;
  case 2:
    y = x;
    if (std::isfinite(x) && x != 0) {
      double lo = std::floor(x), fraction = x - lo;
      y = lo + (fraction > 0.5 || (fraction == 0.5 && std::fmod(lo, 2) != 0));
      y = std::copysign(std::abs(y), x);
    }
    break;
  case 3:
    y = std::floor(x);
    break;
  case 4:
    y = std::sqrt(x);
    break;
  case 5:
    y = 1 / x;
    break;
  case 6:
    y = 1 / std::sqrt(x);
    break;
  case 7:
    y = std::exp2(x);
    break;
  case 9:
    y = x - std::floor(x);
    if (y > double(goc::as_float(0x3f7fffff)))
      y = goc::as_float(0x3f7fffff);
    break;
  case 8:
    y = std::log2(x);
    break;
  }
  float result = float(y);
  if (flush && std::abs(result) < std::ldexp(1.0f, -126))
    result = std::copysign(0.0f, result);
  result = goc_test::omod_f32_reference(result, flags);
  if (flags & GOC_ALU_CLAMP)
    result = std::isnan(result) || result <= 0 ? 0 : std::min(result, 1.0f);
  return result;
}

} // namespace

TEST(Unary, ModifiersMasksAliasesAndCpuLevels) {
  const uint32_t inputs[] = {0,          0x80000000, 0x3f000000, 0xbf000000, 0x3fc00000, 0xbfc00000,
                             0x40200000, 0xc0200000, 0x40600000, 0xc0600000, 0x3f400000, 0xbf400000,
                             0x40800000, 0xc0800000, 0x41800000, 0x3d800000, 1,          0x80000001,
                             0x00800000, 0x80800000, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
                             0x7fc12345, 0xffc12345, 0x4b000001, 0xcb000001, 0x3effffff, 0x3f000001,
                             0x3fffffff, 0x40000001};
  for (int op = 0; op < 11; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned mode = 0; mode < 32; ++mode) {
        uint32_t flags = (mode & 1 ? GOC_ALU_NEG_A : 0) | (mode & 2 ? GOC_ALU_ABS_A : 0) |
                         ((mode >> 2 & 3) << 6) | (mode & 16 ? GOC_ALU_CLAMP : 0);
        for (uint64_t mask : rdna4_exec_masks())
          for (bool alias : {false, true}) {
            SCOPED_TRACE(::testing::Message()
                         << op << "/" << cpu << "/" << mode << "/" << mask << "/" << alias);
            uint32_t a[34], d[34];
            std::fill(a, a + 34, 0xdeadbeef);
            std::fill(d, d + 34, 0xdeadbeef);
            std::copy(inputs, inputs + 32, a + 1);
            uint32_t *pa = a + 1, *pd = alias ? a + 1 : d + 1;
            ASSERT_EQ(functions[op](cpu, mask, flags, &pd, &pa), GOC_SUCCESS);
            for (int lane = 0; lane < 32; ++lane) {
              if (!((mask >> lane) & 1)) {
                EXPECT_EQ(pd[lane], alias ? inputs[lane] : 0xdeadbeef);
                continue;
              }
              float want = reference(op, goc::as_float(inputs[lane]), flags);
              float got = goc::as_float(pd[lane]);
              if (std::isnan(want))
                EXPECT_TRUE(std::isnan(got));
              else if (want == 0 || std::isinf(want))
                EXPECT_EQ(pd[lane], goc::as_bits(want));
              else
                EXPECT_FLOAT_EQ(got, want);
            }
            EXPECT_EQ(a[0], 0xdeadbeef);
            EXPECT_EQ(a[33], 0xdeadbeef);
            EXPECT_EQ(d[0], 0xdeadbeef);
            EXPECT_EQ(d[33], 0xdeadbeef);
          }
      }
}

TEST(Unary, ValidationAndEmptyMask) {
  for (Fn fn : functions) {
    uint32_t a[32] = {}, d[32];
    std::fill(d, d + 32, 0xdeadbeef);
    auto pa = a, pd = d;
    for (uint64_t mask : {UINT64_C(0), UINT64_C(0xffffffff00000000), UINT64_MAX}) {
      EXPECT_EQ(fn(0, mask, GOC_ALU_ABS_B, &pd, &pa), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(0, mask, 1u << 31, &pd, &pa), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(UINT64_C(1) << 63, mask, 0, &pd, &pa), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, mask, 0, &pd, &pa),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
    }
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL, 0, 0, &pd, &pa), GOC_SUCCESS);
    EXPECT_EQ(fn(0, UINT64_C(0xffffffff00000000), GOC_ALU_NEG_A, &pd, &pa), GOC_SUCCESS);
    for (uint32_t value : d)
      EXPECT_EQ(value, 0xdeadbeef);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, 0, &pd, &pa), GOC_SUCCESS);
  }
}

TEST(Unary, FractLiteralBoundaries) {
  const uint32_t cases[][2] = {
      {0, 0},
      {0x80000000, 0},
      {1, 1},
      {0x80000001, 0x3f7fffff},
      {0x80800000, 0x3f7fffff},
      {0x3e800000, 0x3e800000},
      {0xbfa00000, 0x3f400000},
      {0x3f800000, 0},
      {0xbf800000, 0},
      {0xbf7fffff, 0x33800000},
      {0xb3000000, 0x3f7fffff},
      {0x7f7fffff, 0},
      {0xff7fffff, 0},
  };
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (const auto &test : cases) {
      uint32_t a[32], d[32];
      std::fill(a, a + 32, test[0]);
      auto pa = a, pd = d;
      ASSERT_EQ(goc_rdna4_v_fract_f32(cpu, UINT32_MAX, 0, &pd, &pa), GOC_SUCCESS);
      for (uint32_t value : d)
        EXPECT_EQ(value, test[1]) << cpu << "/" << test[0];
    }
}

TEST(Unary, MantissaLiteralSubnormalsAndPassthrough) {
  const uint32_t cases[][2] = {
      {1, 0x3f000000},          {0x80000001, 0xbf000000}, {0x007fffff, 0x3f7ffffe},
      {0x00800000, 0x3f000000}, {0x40c00000, 0x3f400000}, {0, 0},
      {0x80000000, 0x80000000}, {0x7f800000, 0x7f800000}, {0xff800000, 0xff800000},
      {0x7f812345, 0x7f812345}, {0xffc12345, 0xffc12345}};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (const auto &test : cases) {
      uint32_t a[32], d[32];
      std::fill(a, a + 32, test[0]);
      auto pa = a, pd = d;
      ASSERT_EQ(goc_rdna4_v_frexp_mant_f32(cpu, UINT32_MAX, 0, &pd, &pa), GOC_SUCCESS);
      for (uint32_t value : d)
        EXPECT_EQ(value, test[1]);
    }
}

TEST(Unary, HardwareOmodAndMandatoryFlush) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (int op = 0; op < 11; ++op)
      for (unsigned omod = 0; omod < 4; ++omod)
        for (unsigned clamp = 0; clamp < 2; ++clamp)
          for (unsigned neg = 0; neg < 2; ++neg)
            for (uint64_t mask : rdna4_exec_masks())
              for (bool alias : {false, true}) {
                SCOPED_TRACE(::testing::Message() << cpu << '/' << op << '/' << omod << '/' << clamp
                                                  << '/' << neg << '/' << mask << '/' << alias);
                uint32_t a[34], d[34];
                std::fill(a, a + 34, 0xdeadbeef);
                std::fill(d, d + 34, 0xdeadbeef);
                std::copy(goc_test::unary_hardware_inputs, goc_test::unary_hardware_inputs + 32,
                          a + 1);
                uint32_t *pa = a + 1, *pd = alias ? a + 1 : d + 1;
                uint64_t mode = (omod << 6) | (clamp ? GOC_ALU_CLAMP : 0) | neg;
                ASSERT_EQ(functions[op](cpu, mask, mode, &pd, &pa), GOC_SUCCESS);
                for (int lane = 0; lane < 32; ++lane) {
                  if (!((mask >> lane) & 1)) {
                    EXPECT_EQ(pd[lane],
                              alias ? goc_test::unary_hardware_inputs[lane] : 0xdeadbeefu);
                    continue;
                  }
                  uint32_t want = goc_test::unary_hardware[op][omod][clamp][neg][lane];
                  uint32_t magnitude = want & 0x7fffffff;
                  if (magnitude > 0x7f800000) {
                    EXPECT_GT(pd[lane] & 0x7fffffff, 0x7f800000u);
                  } else if (op >= 4 && op <= 8 && magnitude && magnitude < 0x7f800000) {
                    // Transcendental approximation may differ by a few ULPs.
                    // Zero, infinity, sign, and denormal classification must agree.
                    EXPECT_EQ(pd[lane] >> 31, want >> 31);
                    EXPECT_GE(pd[lane] & 0x7fffffff, 0x00800000u);
                    EXPECT_LT(pd[lane] & 0x7fffffff, 0x7f800000u);
                    EXPECT_FLOAT_EQ(goc::as_float(pd[lane]), goc::as_float(want));
                  } else {
                    EXPECT_EQ(pd[lane], want);
                  }
                }
                EXPECT_EQ(a[0], 0xdeadbeefu);
                EXPECT_EQ(a[33], 0xdeadbeefu);
                EXPECT_EQ(d[0], 0xdeadbeefu);
                EXPECT_EQ(d[33], 0xdeadbeefu);
              }
}

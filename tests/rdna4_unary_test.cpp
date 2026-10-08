// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "internal.h"
#include "rdna4_exec_masks.h"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_rdna4_v_log_f32);
const Fn functions[] = {goc_rdna4_v_trunc_f32, goc_rdna4_v_ceil_f32, goc_rdna4_v_rndne_f32,
                        goc_rdna4_v_floor_f32, goc_rdna4_v_sqrt_f32, goc_rdna4_v_rcp_f32,
                        goc_rdna4_v_rsq_f32,   goc_rdna4_v_exp_f32,  goc_rdna4_v_log_f32};

// Independent higher-precision reference, with explicit ties-to-even and
// binary32 rounding before output scaling.
float reference(int op, float input, uint32_t flags) {
  double x = input;
  if (flags & GOC_ALU_ABS_A)
    x = std::abs(x);
  if (flags & GOC_ALU_NEG_A)
    x = -x;
  double y = 0;
  switch (op) {
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
  case 8:
    y = std::log2(x);
    break;
  }
  float result = float(y);
  const float scales[] = {1, 2, 4, 0.5f};
  result *= scales[(flags >> 6) & 3];
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
  for (int op = 0; op < 9; ++op)
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

// SPDX-License-Identifier: MIT

#include "dpp_reference.h"
#include "exec_masks.h"
#include "fuzzy.h"
#include "goc/goc.h"
#include "internal.h"
#include "loose_tolerances.h"
#include "unary_hardware.h"
#include "unary_reference.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {

const decltype(&goc_v_trunc_f32_rdna4_wave64) wave64[] = {
    goc_v_trunc_f32_rdna4_wave64, goc_v_ceil_f32_rdna4_wave64,      goc_v_rndne_f32_rdna4_wave64,
    goc_v_floor_f32_rdna4_wave64, goc_v_sqrt_f32_rdna4_wave64,      goc_v_rcp_f32_rdna4_wave64,
    goc_v_rsq_f32_rdna4_wave64,   goc_v_exp_f32_rdna4_wave64,       goc_v_log_f32_rdna4_wave64,
    goc_v_fract_f32_rdna4_wave64, goc_v_frexp_mant_f32_rdna4_wave64};

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
        for (uint32_t exec_mask : exec_masks())
          for (bool alias : {false, true}) {
            SCOPED_TRACE(::testing::Message()
                         << op << "/" << cpu << "/" << mode << "/" << exec_mask << "/" << alias);
            uint32_t a[34], d[34];
            std::fill(a, a + 34, 0xdeadbeef);
            std::fill(d, d + 34, 0xdeadbeef);
            std::copy(inputs, inputs + 32, a + 1);
            uint32_t *pa = a + 1, *pd = alias ? a + 1 : d + 1;
            ASSERT_EQ(goc_test::unary_functions[op](cpu, exec_mask, flags, &pd, &pa, nullptr),
                      GOC_SUCCESS);
            for (int lane = 0; lane < 32; ++lane) {
              if (!((exec_mask >> lane) & 1)) {
                EXPECT_EQ(pd[lane], alias ? inputs[lane] : 0xdeadbeef);
                continue;
              }
              float want = goc_test::unary_reference(op, goc::as_float(inputs[lane]), flags);
              float got = goc::as_float(pd[lane]);
              if (std::isnan(want))
                EXPECT_TRUE(std::isnan(got));
              else if (want == 0 || std::isinf(want))
                EXPECT_EQ(pd[lane], goc::as_bits(want));
              else
                EXPECT_TRUE(goc_test::fuzzy(got, want, goc_test::unary_f32_tolerance));
            }
            EXPECT_EQ(a[0], 0xdeadbeef);
            EXPECT_EQ(a[33], 0xdeadbeef);
            EXPECT_EQ(d[0], 0xdeadbeef);
            EXPECT_EQ(d[33], 0xdeadbeef);
          }
      }
}

TEST(Unary, ValidationAndEmptyMask) {
  for (auto fn : goc_test::unary_functions) {
    uint32_t a[32] = {}, d[32];
    std::fill(d, d + 32, 0xdeadbeef);
    auto pa = a, pd = d;
    for (uint32_t exec_mask : {0U, UINT32_MAX}) {
      EXPECT_EQ(fn(0, exec_mask, GOC_ALU_ABS_B, &pd, &pa, nullptr), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(0, exec_mask, 1u << 31, &pd, &pa, nullptr), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(1ULL << 63, exec_mask, 0, &pd, &pa, nullptr), GOC_ERROR_INVALID_FLAGS);
      uint64_t unsupported = 2ULL << 16;
      EXPECT_EQ(fn(unsupported | GOC_SEMANTICS_STRICT, exec_mask, 0, &pd, &pa, nullptr),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
    }
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL, 0, 0, &pd, &pa, nullptr), GOC_SUCCESS);
    EXPECT_EQ(fn(0, 0U, GOC_ALU_NEG_A, &pd, &pa, nullptr), GOC_SUCCESS);
    for (uint32_t value : d)
      EXPECT_EQ(value, 0xdeadbeef);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, 0, &pd, &pa, nullptr), GOC_SUCCESS);
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
      ASSERT_EQ(goc_v_fract_f32_rdna4(cpu, UINT32_MAX, 0, &pd, &pa, nullptr), GOC_SUCCESS);
      for (uint32_t value : d)
        EXPECT_EQ(value, test[1]) << cpu << "/" << test[0];
    }
}

TEST(Unary, MantissaLiteralSubnormalsAndSpecialValues) {
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
      ASSERT_EQ(goc_v_frexp_mant_f32_rdna4(cpu, UINT32_MAX, 0, &pd, &pa, nullptr), GOC_SUCCESS);
      for (uint32_t value : d) {
        if ((test[1] & 0x7fffffff) > 0x7f800000) {
          // Loose implementations may preserve or quiet a signaling NaN.
          EXPECT_EQ(value | 0x400000U, test[1] | 0x400000U);
        } else {
          EXPECT_EQ(value, test[1]);
        }
      }
    }
}

TEST(Unary, HardwareOmodAndMandatoryFlush) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (int op = 0; op < 11; ++op)
      for (unsigned omod = 0; omod < 4; ++omod)
        for (unsigned clamp = 0; clamp < 2; ++clamp)
          for (unsigned neg = 0; neg < 2; ++neg)
            for (uint32_t exec_mask : exec_masks())
              for (bool alias : {false, true}) {
                SCOPED_TRACE(::testing::Message()
                             << cpu << '/' << op << '/' << omod << '/' << clamp << '/' << neg << '/'
                             << exec_mask << '/' << alias);
                uint32_t a[34], d[34];
                std::fill(a, a + 34, 0xdeadbeef);
                std::fill(d, d + 34, 0xdeadbeef);
                std::copy(goc_test::unary_hardware_inputs, goc_test::unary_hardware_inputs + 32,
                          a + 1);
                uint32_t *pa = a + 1, *pd = alias ? a + 1 : d + 1;
                uint64_t mode = (omod << 6) | (clamp ? GOC_ALU_CLAMP : 0) | neg;
                ASSERT_EQ(goc_test::unary_functions[op](cpu, exec_mask, mode, &pd, &pa, nullptr),
                          GOC_SUCCESS);
                for (int lane = 0; lane < 32; ++lane) {
                  if (!((exec_mask >> lane) & 1)) {
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
                    EXPECT_TRUE(goc_test::fuzzy(goc::as_float(pd[lane]), goc::as_float(want),
                                                goc_test::unary_f32_tolerance));
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

TEST(Unary, Wave64ModifiersDppSelectorsAndAliases) {

  for (unsigned op = 0; op < 11; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < 32; ++variant)
        for (unsigned route_index = 0; route_index < 8; ++route_index)
          for (unsigned dest = 0; dest < 2; ++dest) {
            uint32_t mode = (variant & 1 ? GOC_ALU_NEG_A : 0) | (variant & 2 ? GOC_ALU_ABS_A : 0) |
                            ((variant >> 2 & 3) << 6) | (variant & 16 ? GOC_ALU_CLAMP : 0);
            uint64_t descriptor = route_index ? goc_test::dpp_modes[route_index - 1] : 0;
            if (descriptor & GOC_DPP16)
              descriptor =
                  (descriptor & ~GOC_DPP_ROW_MASK) | (uint64_t(variant & 15) << GOC_DPP_ROW_SHIFT);
            uint32_t words[2][66], before[2][66];
            for (unsigned reg = 0; reg < 2; ++reg)
              for (unsigned lane = 0; lane < 66; ++lane)
                words[reg][lane] =
                    goc_test::unary_hardware_inputs[(lane + (lane / 32) * 7 + reg * 3) % 32];
            std::memcpy(before, words, sizeof words);
            uint32_t *d = words[dest] + 1;
            const uint32_t *a = words[0] + 1;
            uint64_t exec_mask = 0x87654321fedcba98ULL;
            bool saturate = variant & 1;
            uint32_t exceptions = 0x80000055;
            ASSERT_EQ(wave64[op](cpu | (saturate ? GOC_FP16_OVFL : 0), exec_mask, descriptor | mode,
                                 &d, &a, &exceptions),
                      GOC_SUCCESS);
            EXPECT_EQ(exceptions, 0x80000055U);
            for (unsigned lane = 0; lane < 64; ++lane) {
              int source = int(lane);
              bool enabled = descriptor ? goc_test::dpp_source(descriptor, exec_mask, lane, source)
                                        : ((exec_mask >> lane) & 1);
              if (enabled) {
                float want = goc_test::unary_reference(
                    op, goc::as_float(source < 0 ? 0 : before[0][source + 1]), mode);
                float got = goc::as_float(d[lane]);
                if (std::isnan(want))
                  EXPECT_TRUE(std::isnan(got));
                else if (want == 0 || std::isinf(want))
                  EXPECT_EQ(d[lane], goc::as_bits(want));
                else
                  EXPECT_TRUE(goc_test::fuzzy(got, want, goc_test::unary_f32_tolerance));
              } else {
                EXPECT_EQ(d[lane], before[dest][lane + 1]);
              }
            }
            for (unsigned reg = 0; reg < 2; ++reg) {
              EXPECT_EQ(words[reg][0], before[reg][0]);
              EXPECT_EQ(words[reg][65], before[reg][65]);
            }
          }
}

TEST(Unary, Wave64ValidationAndOptionalState) {
  for (auto fn : wave64) {
    EXPECT_EQ(fn(0, 0, 0, nullptr, nullptr, nullptr), GOC_SUCCESS);
    EXPECT_EQ(fn(0, UINT64_MAX, GOC_ALU_NEG_B, nullptr, nullptr, nullptr), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(
        fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0, nullptr, nullptr, nullptr),
        fn == goc_v_fract_f32_rdna4_wave64 ? GOC_ERROR_UNSUPPORTED_SEMANTICS : GOC_SUCCESS);
    uint32_t exceptions = 0x80000055;
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL, 0, 0, nullptr, nullptr, &exceptions),
              GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
    EXPECT_EQ(exceptions, 0x80000055U);
  }
}

TEST(Unary, Wave64HardwareOmodAndMandatoryFlush) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned op = 0; op < 11; ++op)
      for (unsigned omod = 0; omod < 4; ++omod)
        for (unsigned clamp = 0; clamp < 2; ++clamp)
          for (unsigned neg = 0; neg < 2; ++neg) {
            uint32_t a[64], d[64];
            for (unsigned lane = 0; lane < 64; ++lane)
              a[lane] = goc_test::unary_hardware_inputs[(lane + (lane / 32) * 7) % 32];
            const uint32_t *pa = a;
            uint32_t *pd = d;
            uint64_t mode = (omod << 6) | (clamp ? GOC_ALU_CLAMP : 0) | neg;
            ASSERT_EQ(wave64[op](cpu, UINT64_MAX, mode, &pd, &pa, nullptr), GOC_SUCCESS);
            for (unsigned lane = 0; lane < 64; ++lane) {
              uint32_t want =
                  goc_test::unary_hardware[op][omod][clamp][neg][(lane + (lane / 32) * 7) % 32];
              uint32_t magnitude = want & 0x7fffffff;
              if (magnitude > 0x7f800000) {
                EXPECT_GT(d[lane] & 0x7fffffff, 0x7f800000U);
              } else if (op >= 4 && op <= 8 && magnitude && magnitude < 0x7f800000) {
                EXPECT_EQ(d[lane] >> 31, want >> 31);
                EXPECT_GE(d[lane] & 0x7fffffff, 0x00800000U);
                EXPECT_LT(d[lane] & 0x7fffffff, 0x7f800000U);
                EXPECT_TRUE(goc_test::fuzzy(goc::as_float(d[lane]), goc::as_float(want),
                                            goc_test::unary_f32_tolerance));
              } else {
                EXPECT_EQ(d[lane], want);
              }
            }
          }
}

TEST(Unary, TranscendentalArchitectureGuestPolicies) {
  const decltype(&goc_v_sqrt_f32) narrow[2][5] = {
      {goc_v_sqrt_f32, goc_v_rcp_f32, goc_v_rsq_f32, goc_v_exp_f32, goc_v_log_f32},
      {goc_v_sqrt_f32_rdna4, goc_v_rcp_f32_rdna4, goc_v_rsq_f32_rdna4, goc_v_exp_f32_rdna4,
       goc_v_log_f32_rdna4}};
  const decltype(&goc_v_sqrt_f32_wave64) wide[2][5] = {
      {goc_v_sqrt_f32_wave64, goc_v_rcp_f32_wave64, goc_v_rsq_f32_wave64, goc_v_exp_f32_wave64,
       goc_v_log_f32_wave64},
      {goc_v_sqrt_f32_rdna4_wave64, goc_v_rcp_f32_rdna4_wave64, goc_v_rsq_f32_rdna4_wave64,
       goc_v_exp_f32_rdna4_wave64, goc_v_log_f32_rdna4_wave64}};
  const uint32_t values[] = {0,          0x80000000, 1,          0x80000001, 0x800000,   0x80800000,
                             0x3f800000, 0x40800000, 0x41800000, 0x3e800000, 0x7f800000, 0xff800000,
                             0x7fc00123, 0x7f800001, 0xbf800000, 0x7f7fffff};
  for (unsigned architecture = 0; architecture < 2; ++architecture)
    for (bool wave64 : {false, true})
      for (unsigned op = 0; op < 5; ++op)
        for (unsigned policy = 0; policy < 8; ++policy)
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
            for (unsigned modifier = 0; modifier < 32; ++modifier)
              for (auto descriptor : goc_test::dpp_modes)
                for (bool alias : {false, true}) {
                  uint32_t input[64], output[64], original[64];
                  for (unsigned lane = 0; lane < 64; ++lane) {
                    input[lane] = original[lane] = values[lane % 16];
                    output[lane] = 0xdeadbeef;
                  }
                  uint64_t flags = cpu | (policy & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                                   (policy & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) |
                                   (policy & 4 ? GOC_FP_IEEE_MODE : 0);
                  uint32_t mode = (modifier & 1 ? GOC_ALU_NEG_A : 0) |
                                  (modifier & 2 ? GOC_ALU_ABS_A : 0) | ((modifier >> 2 & 3) << 6) |
                                  (modifier & 16 ? GOC_ALU_CLAMP : 0);
                  uint32_t effective_mode =
                      !architecture && (policy & 4) ? mode & ~GOC_ALU_OMOD_HALF : mode;
                  uint64_t exec_mask = wave64 ? 0x8192a5a571abcdefULL : 0x71abcdef;
                  const uint32_t *a = input;
                  uint32_t *d = alias ? input : output, state = 0x80000001;
                  ASSERT_EQ(wave64 ? wide[architecture][op](flags, exec_mask, mode | descriptor, &d,
                                                            &a, &state)
                                   : narrow[architecture][op](flags, uint32_t(exec_mask),
                                                              mode | descriptor, &d, &a, &state),
                            GOC_SUCCESS);
                  EXPECT_EQ(state, 0x80000001U);
                  for (unsigned lane = 0; lane < (wave64 ? 64U : 32U); ++lane) {
                    int source;
                    if (!goc_test::dpp_source(descriptor, exec_mask, lane, source)) {
                      EXPECT_EQ(d[lane], alias ? original[lane] : 0xdeadbeef);
                      continue;
                    }
                    float want = goc_test::unary_reference(
                        op + 4, goc::as_float(source < 0 ? 0 : original[source]), effective_mode);
                    float got = goc::as_float(d[lane]);
                    if (std::isnan(want)) {
                      EXPECT_TRUE(std::isnan(got));
                    } else if (want == 0 || std::isinf(want)) {
                      EXPECT_EQ(d[lane], goc::as_bits(want));
                    } else {
                      EXPECT_TRUE(goc_test::fuzzy(got, want, goc_test::unary_f32_tolerance));
                    }
                  }
                }
}

TEST(Unary, TranscendentalIeeeScalingWitness) {
  // Unlike RCP_IFLAG, ordinary RCP uses forced-flush OMOD policy even when
  // guest output denormals are preserved. IEEE alone disables RDNA3 scaling.
  uint32_t input[32], output[32];
  std::fill(input, input + 32, 0x40800000); // 4 -> reciprocal 0.25.
  const uint32_t *a = input;
  uint32_t *d = output;
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    ASSERT_EQ(goc_v_rcp_f32(cpu, UINT32_MAX, GOC_ALU_OMOD_2, &d, &a, nullptr), GOC_SUCCESS);
    EXPECT_EQ(d[0], 0x3f000000U);
    ASSERT_EQ(goc_v_rcp_f32(cpu | GOC_FP_IEEE_MODE, UINT32_MAX, GOC_ALU_OMOD_2, &d, &a, nullptr),
              GOC_SUCCESS);
    EXPECT_EQ(d[0], 0x3e800000U);
    ASSERT_EQ(
        goc_v_rcp_f32_rdna4(cpu | GOC_FP_IEEE_MODE, UINT32_MAX, GOC_ALU_OMOD_2, &d, &a, nullptr),
        GOC_SUCCESS);
    EXPECT_EQ(d[0], 0x3f000000U);
  }
}

TEST(Unary, Rdna3Dx10ClampPolicies) {
  const goc_test::UnaryFn narrow[] = {goc_v_trunc_f32, goc_v_ceil_f32,      goc_v_rndne_f32,
                                      goc_v_floor_f32, goc_v_sqrt_f32,      goc_v_rcp_f32,
                                      goc_v_rsq_f32,   goc_v_exp_f32,       goc_v_log_f32,
                                      goc_v_fract_f32, goc_v_frexp_mant_f32};
  const decltype(&goc_v_trunc_f32_wave64) wide[] = {
      goc_v_trunc_f32_wave64, goc_v_ceil_f32_wave64,      goc_v_rndne_f32_wave64,
      goc_v_floor_f32_wave64, goc_v_sqrt_f32_wave64,      goc_v_rcp_f32_wave64,
      goc_v_rsq_f32_wave64,   goc_v_exp_f32_wave64,       goc_v_log_f32_wave64,
      goc_v_fract_f32_wave64, goc_v_frexp_mant_f32_wave64};
  const uint32_t inputs[] = {0x7f800123, 0xff800123, 0x7fc12345, 0xffc12345, 0,          0x80000000,
                             1,          0x80000001, 0x3f000000, 0xbf000000, 0x40000000, 0xc0000000,
                             0x7f800000, 0xff800000, 0x3f800000, 0x7f7fffff};
  for (unsigned op = 0; op < 11; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool precise : {false, true}) {
        if (precise && op == 9) // FRACT has no borrowed exact model.
          continue;
        for (bool wave64 : {false, true})
          for (unsigned policy = 0; policy < 8; ++policy)
            for (unsigned source_mode = 0; source_mode < 8; ++source_mode)
              for (uint64_t route : {uint64_t(0), goc_test::dpp_modes[1]}) {
                uint64_t flags =
                    cpu | (precise ? GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT : 0) |
                    (policy & 1 ? GOC_FP_IEEE_MODE : 0) |
                    (policy & 2 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                    (policy & 4 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0);
                uint64_t mode = route | (source_mode & 1 ? GOC_ALU_NEG_A : 0) |
                                (source_mode & 2 ? GOC_ALU_ABS_A : 0) |
                                (source_mode & 4 ? GOC_ALU_OMOD_2 : 0);
                uint64_t exec_mask = wave64 ? 0x9171abcd8912a5a5ULL : 0x8912a5a5U;
                uint32_t input[64], unclamped[64], actual[64];
                for (unsigned lane = 0; lane < 64; ++lane)
                  input[lane] = inputs[lane % 16];
                auto call = [&](uint64_t f, uint64_t m, uint32_t *d, uint32_t *state,
                                bool alias = false) {
                  const uint32_t *a = alias ? d : input;
                  return wave64 ? wide[op](f, exec_mask, m, &d, &a, state)
                                : narrow[op](f, uint32_t(exec_mask), m, &d, &a, state);
                };
                std::fill(unclamped, unclamped + 64, 0xdeadbeefU);
                ASSERT_EQ(call(flags, mode, unclamped, nullptr), GOC_SUCCESS);
                for (bool alias : {false, true})
                  for (bool preserve : {false, true}) {
                    if (alias)
                      std::copy(input, input + 64, actual);
                    else
                      std::fill(actual, actual + 64, 0xdeadbeefU);
                    uint32_t state = 0x80000055;
                    ASSERT_EQ(call(flags | (preserve ? GOC_FP_DX10_CLAMP_OFF : 0),
                                   mode | GOC_ALU_CLAMP, actual, precise ? nullptr : &state, alias),
                              GOC_SUCCESS);
                    EXPECT_EQ(state, 0x80000055U);
                    for (unsigned lane = 0; lane < 64; ++lane) {
                      if (!((exec_mask >> lane) & 1)) {
                        EXPECT_EQ(actual[lane], alias ? input[lane] : 0xdeadbeefU);
                        continue;
                      }
                      uint32_t value = unclamped[lane], magnitude = value & 0x7fffffffU;
                      if (magnitude > 0x7f800000U && preserve) {
                        if (precise)
                          EXPECT_EQ(actual[lane], value);
                        else
                          EXPECT_GT(actual[lane] & 0x7fffffffU, 0x7f800000U);
                      } else {
                        uint32_t expected = (value >> 31) || magnitude > 0x7f800000U ? 0U
                                            : value > 0x3f800000U                    ? 0x3f800000U
                                                                                     : value;
                        EXPECT_EQ(actual[lane], expected);
                      }
                    }
                  }
              }
      }
  for (unsigned op = 0; op < 11; ++op) {
    EXPECT_EQ(goc_test::unary_functions[op](GOC_FP_DX10_CLAMP_OFF, 1, GOC_ALU_CLAMP, nullptr,
                                            nullptr, nullptr),
              GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(
        wave64[op](GOC_FP_DX10_CLAMP_OFF, 1ULL << 63, GOC_ALU_CLAMP, nullptr, nullptr, nullptr),
        GOC_ERROR_INVALID_FLAGS);
    uint32_t state = 0x12345678;
    EXPECT_EQ(narrow[op](GOC_FP_DX10_CLAMP_OFF | GOC_SEMANTICS_EXACT_EMPIRICAL, 0, UINT64_MAX,
                         nullptr, nullptr, &state),
              GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
    EXPECT_EQ(state, 0x12345678U);
  }
}

TEST(Unary, ExpLogFullRangeModifiersMasksAndAliases) {
  const decltype(&goc_v_exp_f32) narrow[2][2] = {{goc_v_exp_f32, goc_v_log_f32},
                                                 {goc_v_exp_f32_rdna4, goc_v_log_f32_rdna4}};
  const decltype(&goc_v_exp_f32_wave64) wide[2][2] = {
      {goc_v_exp_f32_wave64, goc_v_log_f32_wave64},
      {goc_v_exp_f32_rdna4_wave64, goc_v_log_f32_rdna4_wave64}};
  const uint32_t boundaries[] = {0,           0x80000000U, 1,           0x807fffffU, 0x00800000U,
                                 0x7f7fffffU, 0x7f800000U, 0xff800000U, 0x7fc12345U, 0x7f812345U,
                                 0x3f7fffffU, 0x3f800000U, 0x3f800001U, 0xc2fc0001U, 0xc2fc0000U,
                                 0xc2fbffffU, 0x42ffffffU, 0x43000000U, 0x43000001U};
  for (unsigned arch = 0; arch < 2; ++arch)
    for (unsigned op = 0; op < 2; ++op)
      for (unsigned lanes : {32U, 64U})
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
          for (unsigned mode = 0; mode < 32; ++mode)
            for (unsigned batch = 0; batch < 64; ++batch) {
              const uint64_t flags = cpu | (batch & 1 ? GOC_FP_IEEE_MODE : 0) |
                                     (batch & 2 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                                     (batch & 4 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0);
              const uint32_t modifiers = (mode & 1 ? GOC_ALU_NEG_A : 0) |
                                         (mode & 2 ? GOC_ALU_ABS_A : 0) | ((mode >> 2 & 3) << 6) |
                                         (mode & 16 ? GOC_ALU_CLAMP : 0);
              uint32_t effective = modifiers;
              // These SFUs force flushing; RDNA3 IEEE alone disables OMOD.
              if (!arch && (flags & GOC_FP_IEEE_MODE))
                effective &= ~uint32_t(GOC_ALU_OMOD_HALF);
              uint32_t input[66], output[66], original[64];
              std::fill(input, input + 66, 0xdeadbeefU);
              std::fill(output, output + 66, 0xdeadbeefU);
              for (unsigned lane = 0; lane < lanes; ++lane) {
                const uint32_t index = batch * lanes + lane;
                uint32_t bits = op ? index * 2654435761U
                                   : goc::as_bits(float(int(index) - 2048) * (130.0f / 2048));
                if (batch == 0 && lane < sizeof(boundaries) / sizeof(boundaries[0]))
                  bits = boundaries[lane];
                original[lane] = input[lane + 1] = bits;
              }
              const bool alias = batch & 8;
              uint32_t *d = (alias ? input : output) + 1;
              const uint32_t *a = input + 1;
              const uint64_t exec_mask = batch & 16 ? 0xa55ac33cf00f5aa5ULL : ~0ULL;
              uint32_t state = 0x12345678U;
              ASSERT_EQ(lanes == 32 ? narrow[arch][op](flags, uint32_t(exec_mask), modifiers, &d,
                                                       &a, &state)
                                    : wide[arch][op](flags, exec_mask, modifiers, &d, &a, &state),
                        GOC_SUCCESS);
              EXPECT_EQ(state, 0x12345678U);
              for (unsigned lane = 0; lane < lanes; ++lane) {
                SCOPED_TRACE(::testing::Message() << arch << '/' << op << '/' << lanes << '/' << cpu
                                                  << '/' << mode << '/' << batch << '/' << lane);
                if (!((exec_mask >> lane) & 1)) {
                  EXPECT_EQ(d[lane], alias ? original[lane] : 0xdeadbeefU);
                  continue;
                }
                const float want =
                    goc_test::unary_reference(op + 7, goc::as_float(original[lane]), effective);
                const float got = goc::as_float(d[lane]);
                if (std::isnan(want)) {
                  EXPECT_TRUE(std::isnan(got));
                } else if (want == 0 || std::isinf(want)) {
                  EXPECT_EQ(d[lane], goc::as_bits(want));
                } else {
                  EXPECT_TRUE(goc_test::fuzzy(got, want, goc_test::unary_f32_tolerance));
                }
              }
              EXPECT_EQ(input[0], 0xdeadbeefU);
              EXPECT_EQ(input[lanes + 1], 0xdeadbeefU);
              EXPECT_EQ(output[0], 0xdeadbeefU);
              EXPECT_EQ(output[lanes + 1], 0xdeadbeefU);
            }
}

// Two fixed streams: calibration and independently seeded holdout. Raw words
// span the binary32 exponent/significand space; a second stratum concentrates
// on the finite exp2 range and cancellation near log2(1). Existing literal and
// hardware tests retain classification, zero-sign and exact-output checks.
namespace {

void check_numerical_corpus(uint32_t seed) {
  std::mt19937 random(seed);
  for (unsigned batch = 0; batch < 64; ++batch) {
    uint32_t input[32], output[32];
    for (unsigned lane = 0; lane < 32; ++lane) {
      uint32_t word = random();
      if (batch % 3 == 1)
        word = goc::as_bits(float(int(word % 65537) - 32768) / 256);
      else if (batch % 3 == 2)
        word = 0x3f800000U + (word % 8193) - 4096;
      input[lane] = word;
    }
    const uint32_t *a = input;
    uint32_t *d = output;
    for (unsigned op = 0; op < 11; ++op)
      for (unsigned variant = 0; variant < 32; ++variant) {
        uint32_t mode = (variant & 1 ? GOC_ALU_NEG_A : 0) | (variant & 2 ? GOC_ALU_ABS_A : 0) |
                        ((variant >> 2 & 3) << 6) | (variant & 16 ? GOC_ALU_CLAMP : 0);
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
          ASSERT_EQ(goc_test::unary_functions[op](cpu, UINT32_MAX, mode, &d, &a, nullptr),
                    GOC_SUCCESS);
          for (unsigned lane = 0; lane < 32; ++lane) {
            SCOPED_TRACE(::testing::Message() << seed << '/' << batch << '/' << op << '/' << variant
                                              << '/' << cpu << '/' << lane);
            float expected = goc_test::unary_reference(op, goc::as_float(input[lane]), mode);
            ASSERT_TRUE(goc_test::fuzzy(goc::as_float(output[lane]), expected,
                                        goc_test::unary_f32_tolerance));
          }
        }
      }
  }
}

} // namespace

TEST(Unary, DeterministicNumericalCalibration) { check_numerical_corpus(18731U); }

TEST(Unary, DeterministicNumericalHoldout) { check_numerical_corpus(61357U); }
